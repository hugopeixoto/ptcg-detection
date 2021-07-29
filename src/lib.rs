use image::GenericImageView;
use std::io::BufRead;
use std::io::Write;
use std::time::Instant;
use rayon::prelude::*;

mod sobel;
mod hough;
mod border;
mod corners;

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
    pub sobel: Vec<u32>,
    pub border: Vec<u32>,
    pub hough: Vec<u32>,
    pub source_image: image::DynamicImage,
}

impl ProcessingBuffers {
    pub fn new(width: u32, height: u32) -> Self {
        let mut b = ProcessingBuffers {
            width,
            height,
            sobel: vec![],
            border: vec![],
            hough: vec![],
            source_image: image::DynamicImage::new_rgba8(width, height),
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
    for x in 0..image.width() {
        for y in 0..image.height() {
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
        .unwrap();

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

pub fn process(processing: &mut ProcessingPipeline, dataset: &Vec<DatasetEntry>, stem: &str, debug_images: bool) -> ProcessingTimes {
    let mut times = ProcessingTimes::default();

    if debug_images { processing.buffers.source_image.save(format!("outputs/{}.original.png", stem)).unwrap(); }

    let width = processing.buffers.width;
    let height = processing.buffers.height;

    let mut time = Instant::now();
    sobel::calculate(&processing.frame, width, height, &mut processing.buffers.sobel);
    times.sobel = time.elapsed();
    time = Instant::now();

    border::calculate(&processing.buffers.sobel, width, height, &mut processing.buffers.border);
    times.border = time.elapsed();
    time = Instant::now();

    hough::calculate(&processing.buffers.border, width, height, &mut processing.buffers.hough);
    times.hough = time.elapsed();
    time = Instant::now();

    let corners = corners::calculate(&processing.buffers.hough, width, height);
    times.corners = time.elapsed();
    time = Instant::now();

    if corners.is_empty() {
        return times;
    }

    let a3 = nalgebra::Matrix3::new(
        corners[0].0, corners[1].0, corners[2].0,
        corners[0].1, corners[1].1, corners[2].1,
        1.0, 1.0, 1.0,
    );

    let a1 = nalgebra::Vector3::new(corners[3].0, corners[3].1, 1.0);
    let ax = match a3.lu().solve(&a1) {
        Some(x) => x,
        None => { return times; }
    };
    let a = a3 * nalgebra::Matrix3::new(
        ax[0],   0.0,   0.0,
          0.0, ax[1],   0.0,
          0.0,   0.0, ax[2],
    );

    // 734x1024
    let b3 = nalgebra::Matrix3::new(
        0.0, 734.0,    0.0,
        0.0,  0.0,  1024.0,
        1.0,  1.0,     1.0,
    );

    let b1 = nalgebra::Vector3::new(
         734.0,
        1024.0,
           1.0,
    );

    let bx = b3.lu().solve(&b1).unwrap();
    let b = b3 * nalgebra::Matrix3::new(
        bx[0],   0.0,   0.0,
          0.0, bx[1],   0.0,
          0.0,   0.0, bx[2],
    );

    let c = a * b.try_inverse().unwrap();

    let mut perspective_image = image::DynamicImage::new_rgba8(734, 1024).to_rgba8();
    for y in 0..1024 {
        for x in 0..734 {
            let p = c * nalgebra::Vector3::new(x as f64, y as f64, 1.0);
            let px = (p[0] / p[2]) as i32;
            let py = (p[1] / p[2]) as i32;

            if 0 <= px && px < width as i32 && 0 <= py && py < height as i32 {
                let pixel = processing.buffers.source_image.get_pixel(px as u32, py as u32);
                perspective_image.put_pixel(x, y, pixel);
            }
        }
    }

    times.perspective = time.elapsed();
    time = Instant::now();

    let hasher = img_hash::HasherConfig::new().hash_size(16,16).hash_alg(img_hash::HashAlg::Gradient).to_hasher();

    let hash = hasher.hash_image(&perspective_image);
    let best = dataset.iter().min_by_key(|entry| hash.dist(&entry.hash)).unwrap();

    times.phash = time.elapsed();

    println!("match: {:?} ({})", best.path, hash.dist(&best.hash));

    // perspective_image.save(format!("outputs/{}.perspective.png", stem)).unwrap();
    // image::open(&best.path).unwrap().save(format!("outputs/{}.best.png", stem)).unwrap();
    // println!("[{:?}] debug_images", time.elapsed());

    times
}
