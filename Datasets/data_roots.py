# Datasets for Stereo
STEREO_DR = {'sceneflow':   {'local':   ['/home/amigo/tmp/data/sceneflow', '/home/amigo/tmp/data/sceneflow'],
                            'cluster':  ['/data/datasets/yaoyuh/StereoData/SceneFlow', '/data/datasets/yaoyuh/StereoData/SceneFlow'],
                            'azure':    ['SceneFlow', 'SceneFlow'],
                            'dgx':      ['/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow', '/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/SceneFlow','/ocean/projects/cis210086p/wenshanw/SceneFlow'],
                            }, 
            'tartan':       {'local':   ['/home/amigo/tmp/data/tartan', '/home/amigo/tmp/data/tartan'],
                            'local_test':  ['/peru/tartanair', '/peru/tartanair'],
                            'cluster':  ['/data/datasets/wenshanw/tartan_data', '/data/datasets/wenshanw/tartan_data'],
                            'cluster2':  ['/project/learningvo/tartanair_v1_5', '/project/learningvo/tartanair_v1_5'],
                            'azure':    ['', ''],
                            'dgx':      ['/tmp2/wenshan/tartanair_v1_5', '/tmp2/wenshan/tartanair_v1_5'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/tartanair_v1_5','/ocean/projects/cis210086p/wenshanw/tartanair_v1_5'],
                            },
            'kitti':       {'local':    ['/prague/tartanvo_data/kitti/stereo', '/prague/tartanvo_data/kitti/stereo'], # DEBIG: stereo
                            'cluster':  ['/project/learningvo/stereo_data/kitti/training', '/project/learningvo/stereo_data/kitti/training'],
                            'azure':    ['', ''], # NO KITTI on AZURE yet!!
                            'dgx':      ['/tmp2/wenshan/kitti/training', '/tmp2/wenshan/kitti/training'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/kitti/training','/ocean/projects/cis210086p/wenshanw/kitti/training'],
                            },
            'euroc':       {'local':   ['/prague/tartanvo_data/euroc', '/prague/tartanvo_data/euroc'],
                            },
            }


# Datasets for FlowVo
FLOWVO_DR = {'tartan':      {'local':   '/home/amigo/tmp/data/tartan', # '/home/amigo/tmp/data/tartanair_pose_and_imu',# 
                            'local2':   '/home/amigo/tmp/data/tartanair_pose_and_imu', #'/cairo/tartanair_test_cvpr', # '/home/amigo/tmp/data/tartan', # 
                            'local_test':  '/peru/tartanair',
                            'cluster':  '/data/datasets/wenshanw/tartan_data',
                            'cluster2':  '/project/learningvo/tartanair_v1_5',
                            'azure':    '',
                            'dgx':      '/tmp2/wenshan/tartanair_v1_5',
                            'psc':      '/ocean/projects/cis210086p/wenshanw/tartanair_v1_5',
                            }, 
             'euroc':       {'local':   '/prague/tartanvo_data/euroc', 
                            'cluster2':  '/project/learningvo/euroc',
                            },
             'kitti':       {'local':   '/prague/tartanvo_data/kitti/vo', 
                            },
}

# Datasets for Flow
FLOW_DR =   {'flyingchairs':{'local':   ['/home/amigo/tmp/data/flyingchairs', '/home/amigo/tmp/data/flyingchairs'],
                            'cluster':  ['/project/learningvo/flowdata/FlyingChairs_release', '/project/learningvo/flowdata/FlyingChairs_release'],
                            'azure':    ['FlyingChairs_release', 'FlyingChairs_release'],
                            'dgx':      ['/tmp2/wenshan/flyingchairs', '/tmp2/wenshan/flyingchairs'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/flyingchairs','/ocean/projects/cis210086p/wenshanw/flyingchairs'],
                            }, 
            'flyingthings': {'local':   ['/home/amigo/tmp/data/sceneflow', '/home/amigo/tmp/data/sceneflow/frames_cleanpass'],
                            'cluster':  ['/data/datasets/yaoyuh/StereoData/SceneFlow', '/project/learningvo/flowdata/optical_flow'],
                            'azure':    ['SceneFlow','SceneFlow'],
                            'dgx':      ['/tmp2/DockerTmpfs_yaoyuh/StereoData/SceneFlow', '/tmp2/wenshan/optical_flow'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/SceneFlow','/ocean/projects/cis210086p/wenshanw/optical_flow'],
                            }, 
            'sintel':       {'local':   ['/home/amigo/tmp/data/sintel/training', '/home/amigo/tmp/data/sintel/training'],
                            'cluster':  ['/project/learningvo/flowdata/sintel/training', '/project/learningvo/flowdata/sintel/training'],
                            'azure':    ['sintel/training', 'sintel/training'],
                            'dgx':      ['/tmp2/wenshan/sintel/training', '/tmp2/wenshan/sintel/training'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/sintel/training','/ocean/projects/cis210086p/wenshanw/sintel/training'],
                            }, 
            'tartan':       {'local':   ['/home/amigo/tmp/data/tartan', '/home/amigo/tmp/data/tartan'],
                            'local_test':  ['/peru/tartanair', '/peru/tartanair'],
                            'cluster':  ['/data/datasets/wenshanw/tartan_data', '/data/datasets/wenshanw/tartan_data'],
                            'cluster2':  ['/project/learningvo/tartanair_v1_5', '/project/learningvo/tartanair_v1_5'],
                            'azure':    ['', ''],
                            'dgx':      ['/tmp2/wenshan/tartanair_v1_5', '/tmp2/wenshan/tartanair_v1_5'],
                            'psc':      ['/ocean/projects/cis210086p/wenshanw/tartanair_v1_5','/ocean/projects/cis210086p/wenshanw/tartanair_v1_5'],
                            }, 
            'euroc':        {'local':   ['/prague/tartanvo_data/euroc', '/prague/tartanvo_data/euroc'],
                            'cluster2':  ['/project/learningvo/euroc', '/project/learningvo/euroc'],
                            },
            'kitti':        {'local':   ['/prague/tartanvo_data/kitti/vo', '/prague/tartanvo_data/kitti/vo'],
                            },
    
}

