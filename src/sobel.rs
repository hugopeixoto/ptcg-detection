pub fn calculate(yuyv: &[u8], width: u32, height: u32, sobel: &mut Vec<u32>) {
    for y in 1..height-1 {
        for x in 1..width-1 {
            let val0 = yuyv[( (y - 1)*width*2 + (x - 1)*2) as usize] as i32;
            let val1 = yuyv[( (y - 1)*width*2 + (x + 0)*2) as usize] as i32;
            let val2 = yuyv[( (y - 1)*width*2 + (x + 1)*2) as usize] as i32;
            let val3 = yuyv[( (y + 0)*width*2 + (x - 1)*2) as usize] as i32;

            let val5 = yuyv[( (y + 0)*width*2 + (x + 1)*2) as usize] as i32;
            let val6 = yuyv[( (y + 1)*width*2 + (x - 1)*2) as usize] as i32;
            let val7 = yuyv[( (y + 1)*width*2 + (x + 0)*2) as usize] as i32;
            let val8 = yuyv[( (y + 1)*width*2 + (x + 1)*2) as usize] as i32;

            let gx = -val0 + -2*val3 + -val6 + val2 + 2*val5 + val8;
            let gy = -val0 + -2*val1 + -val2 + val6 + 2*val7 + val8;
            let mag = ((gx*gx + gy*gy) as f64).sqrt().min(255.0);

            sobel[(y*width+x) as usize] = mag as u32;
        }
    }
}
