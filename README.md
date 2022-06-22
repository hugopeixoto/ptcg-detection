# How to run this

I didn't structure this piece of code to run on computers other than my own. It
has a bunch of hardcoded directory paths and requires you to get a dataset from
somewhere. I'll try to document some of the things you'd need in order to run
this.

This assumes you've watched my rustconf 2021 talk
\[[peertube](https://viste.pt/w/mgBRpWnTeBJwuunvvtMi1L)\]
\[[youtube](https://www.youtube.com/watch?v=BLy_YF4nmqQ)\].

## 1. cargo build

After cloning the repo, `cargo build` should work fine. There are three
relevant bins: `cache-dataset`, `photo-detect` and `video-detect`. You'll
probably need to change some paths to make them suit your needs, but it should
build successfully.

If you're not using Linux, this may not compile due to the `v4l` dependency.
This is necessary to use `video-detect`, but if you just want to do the photo
detection instead of video, you can remove this dependency and ignore errors
when building (or just remove) `src/bin/video-detect.rs`.

## 2. card image dataset

The code assumes that you have all the card images in the directory `dataset/`,
in jpg format. You can get them from [pkmncards.com](https://pkmncards.com/) or
[pokemontcg.io](https://pokemontcg.io/) or some other source. There is a
restriction on the image dimensions, mine are all 600x825.

You also need a file named `cardback.jpg` (mine is 585x819) in the root
directory of the project.


## 3. set icon templates

The base detection algorithm is not able to tell apart two similar cards from
two different sets, so I've added a second step where the set symbol is
extracted from the photo and compared with a bunch of set symbol templates.

These are templates are not provided, and they're not easy to create, so the
easiest way to make this run is to edit the `load_templates` function in
`src/lib.rs` and make it return an empty `Vec` instead of what it currently
returns.


## 4. generate cached hashes

The base algorithm uses a perceptual hash to compare images. this is somewhat
expensive to calculate, so the detection programs assume a cache file
`dataset.txt` exists.

To generate this cache, run `cargo run --bin cache-dataset`.


## 5. create an output directory

The `photo-detect` program writes files to the `output/` directory, so you
should create it before running it: `mkdir output`.


## 6. run `photo-detect`

This program takes a bunch of 1920x1080 photos (you need to use this
resolution), analyses them, and writes a bunch of debug images to the `output/`
directory (including a copy of the best three matches found in the dataset).

It reads the photos from the directory `images/canon-1080p/`. You can change
this in `src/bin/photo-detect.rs` and recompile it.


## 7. run `video-detect`

This program reads frames from a camera and analyses them. It assumes that the
camera has a resolution of 1920x1080. It also assumes the camera is available
on `/dev/video3` (`v4l::Device::new(3)`).

It displays the detection on an iced GUI and it also prints the matches to
`stdout`.

