use crate::Luma;

pub fn calculate(image: &dyn Luma<u8>, border: &mut Vec<u32>) {
    let border_threshold = 40;
    let margin = 0u32;

    let width = Luma::<u8>::width(image);
    let height = Luma::<u8>::height(image);

    for i in border.iter_mut() { *i = 0; }

    for x in margin..width - margin {
        for y in margin..height - margin {
            let magnitude = image.get(x, y);
            if magnitude > border_threshold {
                border[(y * width + x) as usize] = 1;
                break;
            }
        }
        for ry in margin..height - margin {
            let y = height - 1 - ry;
            let magnitude = image.get(x, y);
            if magnitude > border_threshold {
                border[(y * width + x) as usize] = 1;
                break;
            }
        }
    }
    for y in margin..height - margin {
        for x in margin..width - margin {
            let magnitude = image.get(x, y);
            if magnitude > border_threshold {
                border[(y * width + x) as usize] = 1;
                break;
            }
        }
        for rx in margin..width - margin {
            let x = width - 1 - rx;
            let magnitude = image.get(x, y);
            if magnitude > border_threshold {
                border[(y * width + x) as usize] = 1;
                break;
            }
        }
    }
}
