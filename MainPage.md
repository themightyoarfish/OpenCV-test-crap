# How to use `solvePnP`
There are currently two branches, `auto_features` which contains an implementation with automatic feature detection and `master` which still uses the manually labeled points. I have not gotten around to merging them and won't attempt it until the automatic detection works.

Due to the different state of the branches, the commands are slightly different.
First, `cd` into the `bin` directory and run `make` or `make solvePnP` for just the PnP executable. 

## General Usage
The `solvePnP` program in the  accepts several command line switches, some of which are required. The `--help` or `-h` flag prints usage info. The program needs calibration data, correspondence information and the images to process. 

### Calibration data
Calibration data is in `calibration/ipad_camera_params.xml` and is given with the `-d` parameter.

### Correspondences
The correspondences are listed in a file with the form 

```
$ cat Data/Bahnhof/correspondence_file_list.txt

../Data/Bahnhof/first_frame_centered.JPG->2.JPG_pts
../Data/Bahnhof/first_frame_centered.JPG->ref_corrected.JPG_pts
../Data/Bahnhof/first_frame_centered.JPG->3.JPG_pts
../Data/Bahnhof/first_frame_centered.JPG->4.JPG_pts
../Data/Bahnhof/first_frame_centered.JPG->5.JPG_pts
../Data/Bahnhof/first_frame_centered.JPG->ref_corrected.JPG_pts

```
notice these paths are so that they resolve correctly when the program is run form the `bin` directory, otherwise they must be adapted. The order is like in the image file list (see below) just without the first frame.

### Images
The image files themselves are likewise given as

```
$ cat Data/Bahnhof/filelist.txt

../Data/Bahnhof/first_frame_centered.JPG
../Data/Bahnhof/2.JPG
../Data/Bahnhof/ref_corrected.JPG
../Data/Bahnhof/3.JPG
../Data/Bahnhof/4.JPG
../Data/Bahnhof/5.JPG
../Data/Bahnhof/ref_corrected.JPG
```

The first three items are first, second and reference frame, the following ones the images for which the relative pose should be computed (ref frame added to see whether we obtain a zero estimate).

### Showing matches
The `-s` switch makes the program display the mathes for each processed image pair (so first, second, ref as well)

## Manual Correspondences
```
$ ./solvePnP -d ../calibration/ipad_camera_params.xml -c ../Data/Gut\ Rosenkrantz/correspondence_file_list.txt -i ../Data/Gut\ Rosenkrantz/filelist.txt
```
will read correspondences and images and print results for the transform first -> ref followed by estimates for each item in `filelist.txt` like so:

```
R_first_ref: [-3.95184, -18.4881, 1.26305]
t_first_ref: [5.069842074737388;
 0.1960446995301588;
 0.5587011246176319]
R: [-1.38975, -8.22568, 0.0995548]
t: [2.569553667534115;
 -0.03211845857978837;
 -0.6013729153047938]
R: [-1.77641, -3.25188, -0.0409188]
t: [1.476273143577587;
 0.005543195530386957;
 -0.3540154827898448]
R: [-0.186572, -4.14644, 0.478872]
t: [0.6638856757474798;
 -0.007209691316604955;
 -0.2036676911316679]
R: [0, 0, 0]
t: [0;
 -5.551115123125783e-17;
 3.33066907387547e-16]
```

## Automatic Correspondences
The newer version in the `auto_features` branch understands more params.
