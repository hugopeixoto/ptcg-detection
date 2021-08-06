use image::GenericImageView;
use image::GenericImage;
use std::io::BufRead;
use std::io::Write;
use std::time::Instant;
use rayon::prelude::*;

pub mod sobel;
pub mod hough;
pub mod border;
pub mod lines;
pub mod corners;
pub mod perspective;
pub mod set_symbol_detection;
pub mod viewer;

pub struct DatasetEntry {
    pub hash: img_hash::ImageHash,
    pub path: std::path::PathBuf,
}

#[derive(Default)]
pub struct ProcessingTimes {
    pub sobel: std::time::Duration,
    pub border: std::time::Duration,
    pub hough: std::time::Duration,
    pub corners: std::time::Duration,
    pub perspective: std::time::Duration,
    pub phash: std::time::Duration,
}

pub struct ProcessingPipeline<'a> {
    pub frame: &'a [u8],
    pub buffers: &'a mut ProcessingBuffers,
}

pub struct ProcessingBuffers {
    pub width: u32,
    pub height: u32,
    pub sobel: Vec<u8>,
    pub border: Vec<u32>,
    pub hough: Vec<u32>,
    pub lines: Vec<(f64, f64, usize)>,
    pub corners: Vec<(f64, f64)>,
    pub source_image: image::DynamicImage,
    pub perspective_image: image::DynamicImage,
}

impl ProcessingBuffers {
    pub fn new(width: u32, height: u32) -> Self {
        let mut b = ProcessingBuffers {
            width,
            height,
            sobel: vec![],
            border: vec![],
            hough: vec![],
            lines: vec![],
            corners: vec![],
            source_image: image::DynamicImage::new_rgba8(width, height),
            perspective_image: image::DynamicImage::new_rgba8(0, 0),
        };

        b.sobel.resize((width * height) as usize, 0);
        b.border.resize((width * height) as usize, 0);
        b.hough.resize(900 * 900, 0);

        b
    }
}


const BUCKET: u8 = 32;
pub fn colors(image: &image::DynamicImage) -> Vec<(u8, u8, u8)> {
    let mut colors: std::collections::HashMap<(u8,u8,u8), u32> = std::collections::HashMap::new();
    for x in 0..GenericImageView::width(image) {
        for y in 0..GenericImageView::height(image) {
            let p = image.get_pixel(x, y);
            let r = p[0] / BUCKET;
            let g = p[1] / BUCKET;
            let b = p[2] / BUCKET;

            if colors.contains_key(&(r,g,b)) {
                let v = colors[&(r,g,b)];
                colors.insert((r,g,b), v + 1);
            } else {
                colors.insert((r,g,b), 1);
            }
        }
    }

    let mut sorted = colors.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|&(_, v)| -(*v as i32));
    sorted.truncate(5);

    return sorted.iter().map(|&(k, _)| *k).collect();
}

pub fn calculate_dataset_entry(path: &std::path::Path) -> DatasetEntry {
    let hasher = img_hash::HasherConfig::new()
        .hash_size(16,16)
        .hash_alg(img_hash::HashAlg::Gradient)
        .to_hasher();

    let file = std::fs::File::open(path).unwrap();
    let img = image::io::Reader::new(std::io::BufReader::new(file))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .blur(1.5);

    DatasetEntry {
        hash: hasher.hash_image(&img),
        path: path.to_path_buf(),
    }
}


pub fn load_or_build_dataset(dataset_path: &str, dataset_cache_filename: &str) -> Vec<DatasetEntry> {
    if std::path::Path::new(dataset_cache_filename).exists() {
        std::io::BufReader::new(std::fs::File::open(dataset_cache_filename).unwrap())
            .lines()
            .filter_map(Result::ok)
            .map(|line| {
                // DatasetEntry.deserialize
                let parts = line.split(' ').collect::<Vec<_>>();

                DatasetEntry {
                        path: std::path::PathBuf::from(&parts[0]),
                        hash: img_hash::ImageHash::from_base64(&parts[1]).unwrap(),
                }
            })
            .collect::<Vec<_>>()
    } else {
        let dataset = std::fs::read_dir(dataset_path)
            .unwrap()
            .filter_map(Result::ok)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|p| calculate_dataset_entry(&p.path()))
            .collect::<Vec<_>>();

        let mut file = std::fs::File::create(dataset_cache_filename).unwrap();
        for entry in dataset.iter() {
            // DatasetEntry.serialize
            file.write(
                format!(
                    "{} {}\n",
                    entry.path.to_str().unwrap(),
                    entry.hash.to_base64(),
                ).as_bytes(),
            ).unwrap();
        }

        dataset
    }
}

pub fn load_templates() -> Vec<(&'static str, f32, image::DynamicImage)> {
    vec![
        ("cpa",   0.20, image::open("templates/cpa.png").unwrap()),
        ("ssh",   0.10, image::open("templates/ssh.png").unwrap()),
        ("exp",   0.10, image::open("templates/exp.png").unwrap()),
        ("gen",   0.18, image::open("templates/gen.png").unwrap()),
        ("pls",   0.15, image::open("templates/pls.png").unwrap()),
        ("lc",    0.10, image::open("templates/lc.png").unwrap()),
        ("promo", 0.20, image::open("templates/promo.png").unwrap()),
        ("bst",   0.20, image::open("templates/bst.png").unwrap()),
        ("sum",   0.10, image::open("templates/sum.png").unwrap()),
    ]
}

pub trait Luma<T> {
    fn get(&self, x: u32, y: u32) -> T;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

struct YUYV442<'a, T> {
    data: &'a [T],
    width: u32,
    height: u32,
}

impl<'a, T> Luma<T> for YUYV442<'a, T> where T: Copy {
    fn get(&self, x: u32, y: u32) -> T {
        self.data[(y * self.width * 2 + x * 2) as usize]
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
}

impl Luma<u8> for image::DynamicImage {
    fn get(&self, x: u32, y: u32) -> u8 {
        self.get_pixel(x, y)[0]
    }
    fn width(&self) -> u32 {
        image::GenericImageView::width(self) as u32
    }
    fn height(&self) -> u32 {
        image::GenericImageView::height(self) as u32
    }
}

struct LumaVec<'a, T> {
    data: &'a [T],
    width: u32,
    height: u32,
}

impl<'a, T> Luma<T> for LumaVec<'a, T> where T: Copy {
    fn get(&self, x: u32, y: u32) -> T {
        self.data[(y * self.width + x) as usize]
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
}

pub fn process<'a>(
    processing: &mut ProcessingPipeline,
    dataset: &'a Vec<DatasetEntry>,
    templates: &Vec<(&'a str, f32, image::DynamicImage)>,
) -> (ProcessingTimes, Option<Vec<(&'a DatasetEntry, u32)>>, Option<&'a str>) {
    let mut times = ProcessingTimes::default();

    let width = processing.buffers.width;
    let height = processing.buffers.height;

    let mut time = Instant::now();
    sobel::calculate(&YUYV442 { data: &processing.frame, width, height }, &mut processing.buffers.sobel);
    times.sobel = time.elapsed();
    time = Instant::now();

    border::calculate(&LumaVec { data: &processing.buffers.sobel, width, height }, &mut processing.buffers.border);
    times.border = time.elapsed();
    time = Instant::now();

    hough::calculate(&processing.buffers.border, width, height, &mut processing.buffers.hough);
    times.hough = time.elapsed();
    time = Instant::now();

    lines::calculate(&processing.buffers.hough, &mut processing.buffers.lines);
    corners::calculate(&processing.buffers.lines, width, height, &mut processing.buffers.corners);
    times.corners = time.elapsed();
    time = Instant::now();

    processing.buffers.perspective_image = image::DynamicImage::new_rgba8(734, 1024);
    if processing.buffers.corners.is_empty() {
        return (times, None, None);
    }

    let buffer = 5;
    let c = perspective::calculate(&processing.buffers.corners, 734.0 + buffer as f64 * 2.0, 1024.0 + buffer as f64 * 2.0);
    if c.is_none() {
        return (times, None, None);
    }

    let c = c.unwrap();

    // TODO: I should build a grayscale image, no need for color.
    // Less work down the line.
    for y in buffer..1024 + buffer {
        for x in buffer..734 + buffer {
            let p = c * nalgebra::Vector3::new(x as f64, y as f64, 1.0);
            let px = (p[0] / p[2]) as i32;
            let py = (p[1] / p[2]) as i32;

            if 0 <= px && px < width as i32 && 0 <= py && py < height as i32 {
                let pixel = processing.buffers.source_image.get_pixel(px as u32, py as u32);
                processing.buffers.perspective_image.put_pixel(x - buffer, y - buffer, pixel);
            }
        }
    }

    times.perspective = time.elapsed();
    time = Instant::now();

    let hasher = img_hash::HasherConfig::new()
        .hash_size(16,16)
        .hash_alg(img_hash::HashAlg::Gradient)
        .to_hasher();

    let hash = hasher.hash_image(&processing.buffers.perspective_image);

    let mut dataset_indexes = (0 .. dataset.len()).collect::<Vec<_>>();
    dataset_indexes.sort_by_key(|i| hash.dist(&dataset[*i].hash));

    let detected_set = detect_set(&processing.buffers.perspective_image.grayscale(), templates);

    times.phash = time.elapsed();

    match detected_set {
        Some(set) => {
            dataset_indexes[0..3].sort_by_key(|i| if dataset[*i].path.to_str().unwrap().contains(set) { 0 } else { 1 });
        },
        None => {}
    }

    (
        times,
        Some(vec![
            (&dataset[dataset_indexes[0]], hash.dist(&dataset[dataset_indexes[0]].hash)),
            (&dataset[dataset_indexes[1]], hash.dist(&dataset[dataset_indexes[1]].hash)),
            (&dataset[dataset_indexes[2]], hash.dist(&dataset[dataset_indexes[2]].hash)),
        ]),
        detected_set,
    )
}

pub fn detect_set<'a>(image: &image::DynamicImage, templates: &Vec<(&'a str, f32, image::DynamicImage)>) -> Option<&'a str> {
    templates
        .par_iter()
        .map(|t| {
            let (score, _) = set_symbol_detection::detect(&image, &t.2, t.1);

            if score != 1.0 {
                Some(t.0)
            } else {
                None
            }
        })
        .find_first(Option::is_some)
        .flatten()
}
