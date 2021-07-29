use detection::*;
use std::time::Instant;

fn main() {
    let dataset = load_or_build_dataset("dataset/", "dataset.txt");

    let mut buffers = ProcessingBuffers::new(1920, 1080);
    let mut frame = vec![];
    frame.resize(1920 * 1080 * 2, 0);

    let paths = std::fs::read_dir("images/canon-1080p/").unwrap();
    for path in paths {
        let filename = path.unwrap().path();
        buffers.source_image = image::open(&filename).expect("failed to read image").resize(1920, 1080, image::imageops::Nearest);

        let luma8 = buffers.source_image.to_luma8();
        buffers.width = luma8.width();
        buffers.height = luma8.height();

        for y in 0 .. luma8.height() {
            for x in 0 .. luma8.width() {
                frame[(y * luma8.width() + x) as usize * 2] = luma8.get_pixel(x, y)[0];
            }
        }

        let mut processing = ProcessingPipeline {
            frame: &frame,
            buffers: &mut buffers,
        };

        let time = Instant::now();

        println!("processing");
        let times = process(
            &mut processing,
            &dataset,
            filename.file_stem().unwrap().to_str().unwrap(),
            false,
        );

        println!("[{:?}] sobel", times.sobel);
        println!("[{:?}] border", times.border);
        println!("[{:?}] hough", times.hough);
        println!("[{:?}] corners", times.corners);
        println!("[{:?}] perspective", times.perspective);
        println!("[{:?}] phash", times.phash);
        println!("[{:?}] processed {:?}", time.elapsed(), filename);
    }
}
