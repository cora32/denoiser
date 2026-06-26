#[derive(Clone)]
pub struct DenoiserParams {
    pub path: String,
    pub block: u32,
    pub delta: i32,
    pub erase_k: u8,
    pub dilate_k: u8,
    pub threshold: u8,
    //
    pub rho: f64,
    pub theta: f64,
    pub line_threshold: i32,
    pub min_line_length: f64,
    pub min_line_gap: f64,
    //
    pub low_threshold: f64,
    pub high_threshold: f64,
    pub aperture: i32,
    pub l2_gradient: bool,
}

impl Default for DenoiserParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            block: 8,
            delta: 10,
            erase_k: 3,
            dilate_k: 1,
            threshold: 1,
            //
            rho: 1.0,                            // rho resolution (pixels)
            theta: std::f64::consts::PI / 180.0, // theta resolution
            line_threshold: 30,                  // threshold
            min_line_length: 20.0,               // min line length
            min_line_gap: 5.0,                   // max line gap
            //
            low_threshold: 50.0,
            high_threshold: 150.0,
            aperture: 3,
            l2_gradient: true,
        }
    }
}
