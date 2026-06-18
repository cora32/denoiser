#[derive(Clone)]
pub struct DenoiserParams {
    pub path: String,
    pub block: u32,
    pub delta: i32,
    pub erase_k: u8,
    pub dilate_k: u8,
    pub threshold: u8,
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
        }
    }
}
