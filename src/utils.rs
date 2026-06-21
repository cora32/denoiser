use anyhow::Result;
use egui::{Color32, ColorImage};
use opencv::{core, imgcodecs, prelude::*};
use opencv::{imgproc, prelude::*};

pub fn mat_to_color_image(src: &core::Mat) -> anyhow::Result<ColorImage> {
    // 1. Ensure the Mat is in RGBA format (OpenCV defaults to BGR)
    let mut rgba_mat = core::Mat::default();
    imgproc::cvt_color(
        src,
        &mut rgba_mat,
        imgproc::COLOR_BGR2RGBA,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // 2. Get dimensions
    let size = rgba_mat.size()?;
    let width = size.width as usize;
    let height = size.height as usize;

    // 3. Extract raw bytes
    let data = rgba_mat.data_bytes()?;

    // 4. Create egui ColorImage
    // data is a flat slice of [r, g, b, a, r, g, b, a, ...]
    let pixels = data
        .chunks_exact(4)
        .map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
        .collect();

    Ok(ColorImage::new([width, height], pixels))
}

pub fn mat_to_png(src: &core::Mat) -> anyhow::Result<Vec<u8>> {
    let mut buf = core::Vector::<u8>::new();
    let params = core::Vector::<i32>::new();

    imgcodecs::imencode(".png", src, &mut buf, &params)?;

    Ok(buf.to_vec())
}
