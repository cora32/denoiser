use egui::epaint::tessellator::Path;
use egui::{Color32, ColorImage};
use image::ColorType::Rgb32F;
use image::{DynamicImage, GrayImage, Luma, imageops};
use image::{GenericImageView, io::Reader as ImageReader};
use image::{ImageBuffer, ImageFormat};
use image::{Rgb, RgbImage};
use image::{Rgba, load_from_memory};
use imageproc::contrast::adaptive_threshold;
use imageproc::distance_transform::Norm;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::drawing::draw_line_segment_mut;
use imageproc::morphology::{dilate, erode};
use imageproc::rect::Rect;
use opencv::photo;
use opencv::prelude::*;
use opencv::{Result, core, imgcodecs, imgproc, prelude::*, types};
use opencv::{core::Mat, prelude::*};
use opencv::{
    core::{Point, Scalar, Vector},
    prelude::*,
};
use std::sync::Arc;
use std::sync::Mutex;

use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::sync::mpsc;
use std::{default, hint};

use crate::ocr_wrapper::{OcrJob, OcrOutput, OcrWorker};
use crate::utils;
use crate::{denoiser_params, ocr_wrapper};

use denoiser_params::DenoiserParams;
use utils::mat_to_color_image;
use utils::mat_to_png;

pub struct DenoiserApp {
    pub tx_orig: mpsc::Sender<Vec<u8>>,
    pub rx_orig: mpsc::Receiver<Vec<u8>>,
    pub tx_denoised: mpsc::Sender<Vec<u8>>,
    pub rx_denoised: mpsc::Receiver<Vec<u8>>,
    pub tx_density: mpsc::Sender<Vec<u8>>,
    pub rx_density: mpsc::Receiver<Vec<u8>>,
    pub tx_lines: mpsc::Sender<ColorImage>,
    pub rx_lines: mpsc::Receiver<ColorImage>,
    pub texture_orig: Option<egui::TextureHandle>,
    pub texture_denoised: Option<egui::TextureHandle>,
    pub texture_denity: Option<egui::TextureHandle>,
    pub texture_lines: Option<egui::TextureHandle>,
    pub params: DenoiserParams,
    pub ocr_worker: Arc<OcrWorker>,
    pub result_holder: Arc<Mutex<Option<Vec<u8>>>>,
}

impl Default for DenoiserApp {
    fn default() -> Self {
        let (tx_orig, rx_orig) = mpsc::channel();
        let (tx_denoised, rx_denoised) = mpsc::channel();
        // let (tx_text, rx_text) = mpsc::channel();
        let (tx_density, rx_density) = mpsc::channel();
        let (tx_lines, rx_lines) = mpsc::channel();
        let ocr_worker = Arc::new(OcrWorker::new());

        Self {
            tx_orig,
            rx_orig,
            tx_denoised,
            rx_denoised,
            tx_density,
            rx_density,
            tx_lines,
            rx_lines,
            texture_orig: None,
            texture_denoised: None,
            texture_denity: None,
            texture_lines: None,
            params: DenoiserParams::default(),
            ocr_worker: ocr_worker,
            result_holder: Arc::new(Mutex::new(None)),
        }
    }
}

impl DenoiserApp {
    pub fn new(_cc: &eframe::CreationContext, file_path: String) -> Self {
        Self {
            params: DenoiserParams {
                path: file_path,
                ..DenoiserParams::default()
            },
            ..DenoiserApp::default()
        }
    }

    fn analyze_density(
        raw_bytes: &[u8],
        user_threshold: u8,
    ) -> (std::vec::Vec<u32>, std::vec::Vec<u8>, u8, i32, i32) {
        println!("Analyzing density...");

        let img = image::load_from_memory(raw_bytes).unwrap();

        let pixels = img.to_luma8();
        let (width, height) = img.dimensions();
        let mut bit_map = vec![0; width as usize];
        let mut min_density = 0;
        let mut max_density = 0;
        let mut threshold_accum: u32 = 0;

        // Get density map
        for x in 0..width {
            for y in 0..height {
                let value = pixels.get_pixel(x, y);

                // Searching for black pixels
                if value[0] < 200 {
                    bit_map[x as usize] += 1;

                    let current_density_value = bit_map[x as usize];

                    if min_density == 0 {
                        min_density = current_density_value;
                    } else if current_density_value < min_density {
                        min_density = current_density_value
                    } else if current_density_value > max_density {
                        max_density = current_density_value
                    }
                }
            }

            threshold_accum += bit_map[x as usize];
        }

        // Calc threshold
        // let threshold: u32 = threshold_accum / width;
        let mut bit_map_copy = bit_map.clone();
        bit_map_copy.sort_unstable();
        let threshold: u32 = bit_map_copy[bit_map_copy.len() / 2 + 10] / 2;

        println!(
            "Width: {} Height: {}; min_density: {} max_density: {}; threshold_accum: {}, threshold: {:#?}",
            width, height, min_density, max_density, threshold_accum, threshold
        );

        // let mut img_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);
        // let mut img_buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
        //     ImageBuffer::from_pixel(width, height, Luma([255]));
        let mut img_buffer = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));

        for x in 0..width {
            let max_y = bit_map[x as usize];

            for y in 0..max_y {
                img_buffer.put_pixel(x, y, Rgb([0, 0, 0]));
            }

            img_buffer.put_pixel(x, threshold, Rgb([255, 0, 0]));

            img_buffer.put_pixel(x, user_threshold as u32, Rgb([0, 255, 0]));
        }

        let mut map_image_bytes: Vec<u8> = Vec::new();
        img_buffer
            .write_to(&mut Cursor::new(&mut map_image_bytes), ImageFormat::Png)
            .unwrap();

        (
            bit_map,
            map_image_bytes,
            threshold as u8,
            height as i32,
            width as i32,
        )
    }

    fn clean_captcha(
        raw_bytes: &[u8],
        block: u32,
        delta: i32,
        erase_k: u8,
        dilate_k: u8,
        density_map: &[u32],
        threshold: u8,
    ) -> Vec<u8> {
        println!(
            "Starting denoiser -> Block: {}; delta: {}; erase_k: {}; dilate_k: {}; ",
            block, delta, erase_k, dilate_k
        );

        // 1. Load the image and convert to Grayscale
        let mut img = image::load_from_memory(raw_bytes)
            .expect("Failed to load image")
            .to_luma8();

        // 2. Scale up: Captchas are usually too small for Tesseract.
        // Making it 2x or 3x larger helps OCR accuracy significantly.
        let (w, h) = img.dimensions();

        //Remove lines by density map
        for x in 0..w {
            let map_value = density_map[x as usize];
            // println!(
            //     "x: {}; map_value: {}; threshold: {}",
            //     x, map_value, threshold
            // );

            if density_map[x as usize] < threshold as u32 {
                for y in 0..h {
                    img.put_pixel(x, y, Luma([255]))
                }
            }
        }

        let img = imageops::resize(&img, w * 2, h * 2, imageops::FilterType::Lanczos3);

        println!("Denoising image w: {}, h: {}", w, h);

        // 3. Adaptive Thresholding: This is better than a fixed threshold for removing
        // background lines that have different color intensities.
        // '8' is the block radius; adjust this based on line thickness.
        let binarized = adaptive_threshold(&img, block, delta);

        // 4. Denoise: Use Erosion then Dilation (Opening) to remove small dots/thin lines
        let denoised = erode(&binarized, Norm::LInf, erase_k);
        let mut final_img = dilate(&denoised, Norm::LInf, dilate_k);

        // 5. Convert back to bytes for Tesseract
        let mut buffer = std::io::Cursor::new(Vec::new());
        final_img
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("Failed to write to buffer");

        // let mut buffer = std::io::Cursor::new(Vec::new());
        // final_img
        //     .write_to(&mut buffer, image::ImageFormat::Png)
        //     .expect("Failed to write to buffer");

        println!("Denoising complete.");

        buffer.into_inner()
    }

    fn detect_lines(
        png_bytes: Vec<u8>,
        //
        rho: f64,
        theta: f64,
        line_threshold: i32,
        min_line_length: f64,
        min_line_gap: f64,
        //
        low_threshold: f64,
        high_threshold: f64,
        aperture: i32,
        l2_gradient: bool,
        inpaint_radius: f64,
    ) -> (ColorImage, Vec<u8>) {
        // 1. Read the PNG file bytes (Compressed)
        // let png_bytes = fs::read(path).expect("Unable to read file");
        // let resized = Self::resize(&png_bytes, 400, 200);

        // 2. Decode the bytes into a raw pixel Mat (BGR or Grayscale)
        let buf = Vector::<u8>::from_iter(png_bytes);
        let src = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_GRAYSCALE).unwrap();

        let mut fixed_aperture = aperture;
        if aperture % 2 == 0 {
            fixed_aperture = (aperture + 1).min(7);
        }

        println!("low_threshold: {}", low_threshold);

        // let mut blurred = core::Mat::default();

        // imgproc::gaussian_blur(
        //     &src,
        //     &mut blurred,
        //     opencv::core::Size::new(5, 5),
        //     10.0,
        //     10.0,
        //     opencv::core::BORDER_DEFAULT,
        //     opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        // )
        // .unwrap();

        // 3. Create a binary edge map (HoughLinesP requires this)
        let mut edges = core::Mat::default();
        imgproc::canny(
            &src,
            &mut edges,
            low_threshold,  // Low threshold
            high_threshold, // High threshold
            fixed_aperture, // Aperture size
            l2_gradient,
        )
        .unwrap();

        let mut lines = Vector::<opencv::core::Vec4i>::new();

        imgproc::hough_lines_p(
            &edges,          // input binary image
            &mut lines,      // output lines
            rho,             // rho resolution (pixels)
            theta,           // theta resolution
            line_threshold,  // threshold
            min_line_length, // min line length
            min_line_gap,    // max line gap
        )
        .unwrap();

        // Get only horizontal lines
        let mut horizontal_lines = Vec::<opencv::core::Vec4i>::new();
        let mut mask = Mat::zeros(src.rows(), src.cols(), opencv::core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();

        for line in lines {
            let dx = (line[2] - line[0]) as f64;
            let dy = (line[3] - line[1]) as f64;

            let angle = dy.atan2(dx).to_degrees();

            if angle.abs() < 10.0 {
                // Fills mask
                imgproc::line(
                    &mut mask,
                    Point::new(line[0], line[1]),
                    Point::new(line[2], line[3]),
                    Scalar::all(255.0),
                    3,
                    imgproc::LINE_AA,
                    0,
                )
                .unwrap();

                // Adds data to array to draw on next step
                horizontal_lines.push(line);
            }
        }

        println!("Detected {} lines", horizontal_lines.len());

        println!("Inpainting...");
        let mut result = Mat::default();

        println!("src size: {:?}", src.size().unwrap());
        println!("mask size: {:?}", mask.size().unwrap());
        println!("result size: {:?}", result.size().unwrap());

        photo::inpaint(
            &src,
            &mask,
            &mut result,
            inpaint_radius,
            photo::INPAINT_TELEA,
        )
        .unwrap();

        (
            mat_to_color_image(&mask).unwrap(),
            mat_to_png(&result).unwrap(),
        )
    }

    fn resize(bytes: &[u8], new_width: u32, new_height: u32) -> Vec<u8> {
        let img = load_from_memory(bytes).expect("Failed to load image from memory");
        let resized = img.resize(new_width, new_height, imageops::FilterType::Lanczos3);
        let mut buffer = Cursor::new(Vec::new());
        resized
            .write_to(&mut buffer, ImageFormat::Png)
            .expect("Failed to write to buffer");

        buffer.into_inner()
    }

    fn add_white_lines(
        bytes: &Vec<u8>,
        hl_thickness: u32,
        hl_step: i32,
        vl_thickness: u32,
        vl_step: i32,
    ) -> DynamicImage {
        let mut img = load_from_memory(bytes)
            .expect("Failed to decode image")
            .into_rgba8();

        let color = Rgba([255, 255, 255, 255]);
        let (width, height) = img.dimensions();

        // Horizontal lines
        for y in (0..height).step_by(hl_step as usize) {
            // let start = (0.0, y as f32);
            // let end = (width as f32, y as f32);

            let rect = Rect::at(0, y as i32).of_size(width, hl_thickness);

            //draw_line_segment_mut(&mut img, start, end, color);
            draw_filled_rect_mut(&mut img, rect, color);
        }

        // Horizontal lines
        for y in (0..height).step_by(hl_step as usize) {
            // let start = (0.0, y as f32);
            // let end = (width as f32, y as f32);

            let rect = Rect::at(0, y as i32).of_size(width, hl_thickness);

            //draw_line_segment_mut(&mut img, start, end, color);
            draw_filled_rect_mut(&mut img, rect, color);
        }

        // Vertical lines
        for x in (0..width).step_by(vl_step as usize) {
            let rect = Rect::at(x as i32, 0).of_size(vl_thickness, height);

            //draw_line_segment_mut(&mut img, start, end, color);
            draw_filled_rect_mut(&mut img, rect, color);
        }

        DynamicImage::ImageRgba8(img)
    }

    fn img_to_bytes(img: &DynamicImage) -> Vec<u8> {
        let mut result_bytes = Vec::new();
        let mut cursor = Cursor::new(&mut result_bytes);

        img.write_to(&mut cursor, image::ImageFormat::Png)
            .expect("Failed to encode PNG");

        result_bytes
    }

    pub fn denoise(&mut self) {
        let tx_orig = self.tx_orig.clone();
        let tx_denoised = self.tx_denoised.clone();
        let tx_density = self.tx_density.clone();
        let tx_lines = self.tx_lines.clone();
        let params = self.params.clone();
        let result_holder_handler = self.result_holder.clone();

        println!("Loading {}", params.path);

        tokio::spawn(async move {
            let bytes = fs::read(params.path.clone()).expect("Unable to read file");

            let resized = Self::resize(&bytes.clone(), 400, 200);

            tx_orig.send(resized.clone()).unwrap();

            let white_lines = Self::add_white_lines(
                &resized,
                params.hl_thickness,
                params.hl_step,
                params.vl_thickness,
                params.vl_step,
            );

            // Getting density
            // let (bit_map, density_bytes, threshold, height, width) =
            //     Self::analyze_density(&resized, params.threshold);
            // tx_density.send(density_bytes).unwrap();

            // println!("Width: {} Height: {}", width, height);

            let bytes = Self::img_to_bytes(&white_lines);

            //Line detection
            println!("Detecting lines...");
            let (lines_mask, inpainted) = Self::detect_lines(
                bytes.clone(),
                //
                params.rho,
                params.theta,
                params.line_threshold,
                params.min_line_length,
                params.min_line_gap,
                //
                params.low_threshold,
                params.high_threshold,
                params.aperture,
                params.l2_gradient,
                params.inpaint_radius,
            );
            // let img_with_lines_bytes = draw_lines();
            tx_lines.send(lines_mask).unwrap();

            // Denoising
            // println!("Denoising...");
            // let denoised_bytes = Self::clean_captcha(
            //     &inpainted.clone(),
            //     params.block,
            //     params.delta,
            //     params.erase_k,
            //     params.dilate_k,
            //     &bit_map,
            //     params.threshold,
            // );
            tx_denoised.send(inpainted.clone()).unwrap();

            if let Ok(mut lock) = result_holder_handler.lock() {
                *lock = Some(inpainted.clone());
            }

            tx_density.send(bytes.clone()).unwrap();

            println!("Done");
        });
    }

    pub fn ocr(&mut self) {
        if let Ok(lock) = self.result_holder.lock() {
            if let Some(bytes) = &*lock {
                self.ocr_worker
                    .tx
                    .send(OcrJob {
                        bytes: (*bytes).clone(),
                    })
                    .unwrap();
            }
        }
    }

    pub fn load_texture_from_bytes(
        ui: &mut egui::Ui,
        bytes: Vec<u8>,
        texture: &mut Option<egui::TextureHandle>,
    ) {
        let image = image::load_from_memory(&bytes).unwrap().to_rgba8();

        let size = [image.width() as usize, image.height() as usize];

        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        Self::load_texture_from_color_image(ui, color_image, texture)
    }

    pub fn load_texture_from_color_image(
        ui: &mut egui::Ui,
        color_image: ColorImage,
        texture: &mut Option<egui::TextureHandle>,
    ) {
        *texture = Some(ui.load_texture("data", color_image, Default::default()));
    }
}
