use image::GenericImageView;

#[inline]
fn score(image: &image::DynamicImage, template: &image::DynamicImage, x: u32, y: u32, template_total_squared: f32) -> f32 {
    let mut score = 0f32;

    let mut totes_image = 0f32;

    for dy in 0 .. template.height() {
        for dx in 0 .. template.width() {
            let template_pixel = unsafe { template.unsafe_get_pixel(dx, dy) };

            if template_pixel[3] != 0 {
                let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                let template_value = template_pixel[0] as f32;

                totes_image += image_value.powf(2.0);
                score += (image_value - template_value).powf(2.0);
            }
        }
    }

    score / (totes_image * template_total_squared).sqrt()
}

pub fn detect(image: &image::DynamicImage, template: &image::DynamicImage, threshold: f32) -> (f32, image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) {
    let mut marked = image.clone().to_rgba8();

    let mut template_total_squared = 0f32;
    for dy in 0 .. template.height() {
        for dx in 0 .. template.width() {
            let template_pixel = unsafe { template.unsafe_get_pixel(dx, dy) };
            if template_pixel[3] != 0 {
                let template_value = template_pixel[0] as f32;
                template_total_squared += template_value.powf(2.0);
            }
        }
    }
    let mut positions = vec![];

    for y in 900..image.height().min(1000) - template.height() {
        for x in 10..image.width().min(130) - template.width() {
            positions.push((x, y));
        }
    }

    for y in 850..image.height() - template.height() {
        for x in 600..image.width() - template.width() {
            positions.push((x, y));
        }
    }

    for y in 500..image.height().min(620) - template.height() {
        for x in 600..image.width() - template.width() {
            positions.push((x, y));
        }
    }

    let mut best_score = 1.0f32;
    for &(x, y) in positions.iter() {
        let p = score(image, template, x, y, template_total_squared);
        if p < threshold {
            best_score = best_score.min(p);
            for k in 0..template.width(){
                marked.put_pixel(x+k, y, image::Rgba([255, 0, 255, 255]));
                marked.put_pixel(x+k, y+template.height()-1, image::Rgba([255, 0, 255, 255]));
            }
            for k in 0..template.height() {
                marked.put_pixel(x, y+k, image::Rgba([255, 0, 255, 255]));
                marked.put_pixel(x+template.width()-1, y+k, image::Rgba([255, 0, 255, 255]));
            }
        }
    }

    (best_score, marked)
}
