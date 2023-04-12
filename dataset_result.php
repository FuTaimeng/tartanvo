<?php
ob_start(); // Start output buffering
?>
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Result</title>
</head>
<body>
    <h1>Dataset Result</h1>
    <h2>KITTI</h2>
    <?php
    for ($i = 0; $i <= 10; $i++) {
        if ($i == 3) {
            // skip trajectory 3
            continue;
        }
        echo '<h3>Trajectory  '.$i.'</h3>';
        echo '<div style="display:flex;">';
        echo '<figure>';
        echo '<img src="/results/kitti_'.$i.'_val_aligned.png" alt="kitti_'.$i.'_val_aligned.png">';
        echo '<figcaption style="text-align: center;">Aligned</figcaption>';
        echo '</figure>';
        

        echo '<figure>';
        
        // echo '<img src="/results/kitti_'.$i.'_val_origin.png" alt="kitti_'.$i.'_val_origin.png">';
        echo '<img src="/results/kitti_'.$i.'_val_aligned_scaled.png" alt="kitti_'.$i.'_val_aligned_scaled.png">';
        echo '<figcaption style="text-align: center;">Aligned and Scaled</figcaption>';
        echo '</figure>';
        echo '</div>';
    }
    ?>
    
    <br><br><br>
    <h2>EuRoc</h2>
    <?php
    for ($i = 0; $i <= 10; $i++) {
        echo '<h3>Trajectory  '.$i.'</h3>';
        echo '<div style="display:flex;">';

        // echo '<img src="/results/euroc_'.$i.'_val_origin.png" alt="euroc_'.$i.'_val_origin.png">';
        echo '<figure>';
        echo '<img src="/results/euroc_'.$i.'_val_aligned_scaled.png" alt="euroc_'.$i.'_val_aligned_scaled.png">';
        echo '<figcaption style="text-align: center;">Aligned</figcaption>';
        echo '</figure>';

        echo '<figure>';
        echo '<img src="/results/euroc_'.$i.'_val_aligned.png" alt="euroc_'.$i.'_val_aligned.png">';
        echo '<figcaption style="text-align: center;">Aligned and Scaled </figcaption>';
        echo '</figure>';
        echo '</div>';
    }
    ?>

</body>
</html>
<?php
$content = ob_get_clean(); // Get the buffered output and clean the buffer
file_put_contents('dataset_result.html', $content); // Save the output to a file
?>
