pub fn calculate(corners: &Vec<(f64, f64)>, width: f64, height: f64) -> nalgebra::Matrix3<f64> {
    let a3 = nalgebra::Matrix3::new(
        corners[0].0, corners[1].0, corners[2].0,
        corners[0].1, corners[1].1, corners[2].1,
        1.0, 1.0, 1.0,
    );

    let a1 = nalgebra::Vector3::new(corners[3].0, corners[3].1, 1.0);
    let ax = a3.lu().solve(&a1).unwrap();

    let a = a3 * nalgebra::Matrix3::new(
        ax[0],   0.0,   0.0,
          0.0, ax[1],   0.0,
          0.0,   0.0, ax[2],
    );

    let b3 = nalgebra::Matrix3::new(
        0.0, width,    0.0,
        0.0,  0.0,  height,
        1.0,  1.0,     1.0,
    );

    let b1 = nalgebra::Vector3::new(
         width,
        height,
           1.0,
    );

    let bx = b3.lu().solve(&b1).unwrap();
    let b = b3 * nalgebra::Matrix3::new(
        bx[0],   0.0,   0.0,
          0.0, bx[1],   0.0,
          0.0,   0.0, bx[2],
    );

    a * b.try_inverse().unwrap()
}
