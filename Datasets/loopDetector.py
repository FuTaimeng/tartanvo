import numpy as np
from .transformation import pos_quats2SE_matrices
import cv2
from scipy.spatial.transform import Rotation


def is_pose_approximate(T1, T2, trans_th, rot_th):
    zero = np.array([0, 0, 0, 1]).reshape(4, 1)
    z1 = np.array([0, 0, 1, 1]).reshape(4, 1)
    trans_err = np.linalg.norm(np.dot(T1, zero) - np.dot(T2, zero))
    rot_err = np.rad2deg(np.arccos(np.dot(np.dot(T1[0:3, 0:3], z1[0:3, :]).T, np.dot(T2[0:3, 0:3], z1[0:3, :]))))
    return trans_err <= trans_th and rot_err <= rot_th

def gt_pose_loop_detector(poses, loop_min_interval, trans_th, rot_th):
    N = len(poses)
    Ts = pos_quats2SE_matrices(poses)

    links = []
    for i in range(N-1):
        links.append((i, i+1))

    i = 0
    while i < N:
        flag = False
        j = i + loop_min_interval
        while j < N:
            if is_pose_approximate(Ts[i], Ts[j], trans_th, rot_th):
                links.append((i, j))
                flag = True
                # print('loop edge:', (i, j))
                # j += 10
                j += 1
            else:
                j += 1
        if flag:
            # i += 10
            i += 1
        else:
            i += 1
    # print(links)
    print('With trans_th={}, rot_th={}, find {} links!'.format(trans_th, rot_th, len(links)))

    return links


def bow_orb_loop_detector(images, loop_min_interval, voc_dir='ORBvoc.txt'):
    import pyDBoW3 as bow

    N = len(images)

    links = []
    for i in range(N-1):
        links.append((i, i+1))

    voc = bow.Vocabulary()
    voc.load(voc_dir)
    db = bow.Database()
    db.setVocabulary(voc)
    del voc

    # extract features using OpenCV
    features_list = []
    orb = cv2.ORB_create()
    for img in images:
        features_list.append(orb.detect(img))

    # add features to database
    for features in features_list:
        db.add(features)

    # query features
    feature_to_query = 1
    results = db.query(features_list[feature_to_query])
    print(results)

    exit(0)

    del db
    return links


def adj_loop_detector(N, loop_range, loop_interval):
    links = []
    for i in range(N-1):
        links.append((i, i+1))
    for i in range(0, N-loop_range, loop_interval):
        links.append((i, i+loop_range))
    return links


def generate_g2o(output_fname, poses, motions, links):
    g2o_vertex = 'VERTEX_SE3:QUAT {id} {tx} {ty} {tz} {qi} {qj} {qk} {qw} '
    g2o_edge = 'EDGE_SE3:QUAT {id1} {id2} {tx} {ty} {tz} {qi} {qj} {qk} {qw} {I11} {I12} {I13} {I14} {I15} {I16} {I22} {I23} {I24} {I25} {I26} {I33} {I34} {I35} {I36} {I44} {I45} {I46} {I55} {I56} {I66} '
    g2o_edge_noinfo = 'EDGE_SE3:QUAT {id1} {id2} {tx} {ty} {tz} {qi} {qj} {qk} {qw} 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1 '
    g2o_edge_uniinfo = 'EDGE_SE3:QUAT {id1} {id2} {tx} {ty} {tz} {qi} {qj} {qk} {qw} {It} 0 0 0 0 0 {It} 0 0 0 0 {It} 0 0 0 {Ir} 0 0 {Ir} 0 {Ir} '

    g2o_file = ''

    try:
        poses = np.loadtxt(poses)
        motions = np.loadtxt(motions)
        links = np.loadtxt(links)
    except:
        pass

    N = len(poses)
    M = len(links)

    min_idx = np.min(links[len(poses)-1:])
    max_idx = np.max(links[len(poses)-1:])

    for i in range(min_idx, max_idx+1):
        t = poses[i, :3]
        q = poses[i, 3:]
        g2o_file += g2o_vertex.format(id=i, tx=t[0], ty=t[1], tz=t[2], qi=q[0], qj=q[1], qk=q[2], qw=q[3]) + '\n'

    for i in range(M):
        id1 = links[i][0]
        id2 = links[i][1]
        if not (id1>=min_idx and id1<=max_idx and id2>=min_idx and id2<=max_idx):
            continue
        id1 -= min_idx
        id2 -= min_idx
        t = motions[i, :3]
        q = motions[i, 3:]
        # noinfo
        g2o_file += g2o_edge_noinfo.format(id1=id1, id2=id2, tx=t[0], ty=t[1], tz=t[2], qi=q[0], qj=q[1], qk=q[2], qw=q[3]) + '\n'

        # uniinfo
        # It = 1.0 / max(np.linalg.norm(t), 0.01) * 0.1
        # Ir = 1.0 / max(np.rad2deg(Rotation.from_quat(q).magnitude()), 0.1)
        # # print(It, Ir)
        # g2o_file += g2o_edge_uniinfo.format(id1=id1, id2=id2, tx=t[0], ty=t[1], tz=t[2], qi=q[0], qj=q[1], qk=q[2], qw=q[3], It=It, Ir=Ir) + '\n'

    with open(output_fname, 'w') as f:
        f.write(g2o_file)
