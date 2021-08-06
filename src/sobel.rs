use crate::Luma;

pub fn calculate(image: &dyn Luma<u8>, sobel: &mut Vec<u8>) {
    for y in 1..Luma::<u8>::height(image) - 1 {
        for x in 1..Luma::<u8>::width(image) - 1 {
            let val0 = image.get(x - 1, y - 1) as i32;
            let val1 = image.get(x + 0, y - 1) as i32;
            let val2 = image.get(x + 1, y - 1) as i32;

            let val3 = image.get(x - 1, y + 0) as i32;
            let val5 = image.get(x + 1, y + 0) as i32;

            let val6 = image.get(x - 1, y + 1) as i32;
            let val7 = image.get(x + 0, y + 1) as i32;
            let val8 = image.get(x + 1, y + 1) as i32;

            let gx = -val0 + -2*val3 + -val6 + val2 + 2*val5 + val8;
            let gy = -val0 + -2*val1 + -val2 + val6 + 2*val7 + val8;
            let mag = ((gx*gx + gy*gy) as f64).sqrt().min(255.0) as u8;

            sobel[(y * Luma::<u8>::width(image) + x) as usize] = mag;
        }
    }
}
