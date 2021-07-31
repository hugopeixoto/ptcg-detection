pub fn calculate(sobel: &Vec<u32>, width: u32, height: u32, border: &mut Vec<u32>) {
    let border_threshold = 40;
    let margin = 0;

    for i in border.iter_mut() { *i = 0; }

    for x in margin..width as usize - margin {
        for y in margin..height as usize - margin {
            let magnitude = sobel[y * width as usize + x];
            if magnitude > border_threshold {
                border[y * width as usize + x] = 1;
                break;
            }
        }
        for ry in margin..height as usize - margin {
            let y = height as usize - 1 - ry;
            let magnitude = sobel[y * width as usize + x];
            if magnitude > border_threshold {
                border[y * width as usize + x] = 1;
                break;
            }
        }
    }
    for y in margin..height as usize - margin {
        for x in margin..width as usize - margin {
            let magnitude = sobel[(y * width as usize + x) as usize];
            if magnitude > border_threshold {
                border[y * width as usize + x] = 1;
                break;
            }
        }
        for rx in margin..width as usize - margin {
            let x = width as usize - 1 - rx;
            let magnitude = sobel[(y * width as usize + x) as usize];
            if magnitude > border_threshold {
                border[y * width as usize + x] = 1;
                break;
            }
        }
    }
}
