use eframe::egui;
use egui::epaint::tessellator::Path;
use egui::{Color32, ColorImage};
use image::ColorType::Rgb32F;
use image::{DynamicImage, GrayImage, Luma, imageops};
use image::{GenericImageView, io::Reader as ImageReader};
use image::{ImageBuffer, ImageFormat};
use image::{Rgb, RgbImage};
use imageproc::contrast::adaptive_threshold;
use imageproc::distance_transform::Norm;
use imageproc::morphology::{dilate, erode};
use opencv::photo;
use opencv::prelude::*;
use opencv::{Result, core, imgcodecs, imgproc, prelude::*, types};
use opencv::{core::Mat, prelude::*};
use opencv::{
    core::{Point, Scalar, Vector},
    prelude::*,
};

use std::default;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::sync::mpsc;

mod denoiser_params;
mod utils;

use denoiser_params::DenoiserParams;
use utils::mat_to_color_image;
use utils::mat_to_png;

struct DenoiserApp {
    tx_orig: mpsc::Sender<Vec<u8>>,
    rx_orig: mpsc::Receiver<Vec<u8>>,
    tx_denoised: mpsc::Sender<Vec<u8>>,
    rx_denoised: mpsc::Receiver<Vec<u8>>,
    tx_density: mpsc::Sender<Vec<u8>>,
    rx_density: mpsc::Receiver<Vec<u8>>,
    tx_lines: mpsc::Sender<ColorImage>,
    rx_lines: mpsc::Receiver<ColorImage>,
    tx_text: mpsc::Sender<Vec<u8>>,
    rx_text: mpsc::Receiver<Vec<u8>>,
    texture_orig: Option<egui::TextureHandle>,
    texture_denoised: Option<egui::TextureHandle>,
    texture_denity: Option<egui::TextureHandle>,
    texture_lines: Option<egui::TextureHandle>,
    params: DenoiserParams,
}

impl Default for DenoiserApp {
    fn default() -> Self {
        let (tx_orig, rx_orig) = mpsc::channel();
        let (tx_denoised, rx_denoised) = mpsc::channel();
        let (tx_text, rx_text) = mpsc::channel();
        let (tx_density, rx_density) = mpsc::channel();
        let (tx_lines, rx_lines) = mpsc::channel();

        Self {
            tx_orig,
            rx_orig,
            tx_denoised,
            rx_denoised,
            tx_density,
            rx_density,
            tx_lines,
            rx_lines,
            tx_text,
            rx_text,
            texture_orig: None,
            texture_denoised: None,
            texture_denity: None,
            texture_lines: None,
            params: DenoiserParams::default(),
        }
    }
}

impl DenoiserApp {
    fn new(_cc: &eframe::CreationContext, file_path: String) -> Self {
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
        path: String,
        height: i32,
        width: i32,
        rho: f64,
        theta: f64,
        line_threshold: i32,
        min_line_length: f64,
        min_line_gap: f64,
    ) -> (ColorImage, Vec<u8>) {
        // 1. Read the PNG file bytes (Compressed)
        let png_bytes = fs::read(path).expect("Unable to read file");

        // 2. Decode the bytes into a raw pixel Mat (BGR or Grayscale)
        let buf = Vector::<u8>::from_iter(png_bytes);
        let src = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_GRAYSCALE).unwrap();

        // 3. Create a binary edge map (HoughLinesP requires this)
        let mut edges = core::Mat::default();
        imgproc::canny(
            &src, &mut edges, 50.0,  // Low threshold
            150.0, // High threshold
            3,     // Aperture size
            false,
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
        let mut mask = Mat::zeros(height, width, opencv::core::CV_8UC1)
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

        photo::inpaint(&src, &mask, &mut result, 3.0, photo::INPAINT_TELEA).unwrap();

        (
            mat_to_color_image(&mask).unwrap(),
            mat_to_png(&result).unwrap(),
        )
    }

    fn denoise(&mut self) {
        let tx_orig = self.tx_orig.clone();
        let tx_denoised = self.tx_denoised.clone();
        let tx_density = self.tx_density.clone();
        let tx_lines = self.tx_lines.clone();
        let params = self.params.clone();

        println!("Loading {}", params.path);

        tokio::spawn(async move {
            let bytes = fs::read(params.path.clone()).expect("Unable to read file");

            tx_orig.send(bytes.clone()).unwrap();

            // Getting density
            let (bit_map, density_bytes, threshold, height, width) =
                Self::analyze_density(&bytes, params.threshold);
            tx_density.send(density_bytes).unwrap();

            //Line detection
            println!("Detecting lines...");
            let (lines_mask, inpainted) = Self::detect_lines(
                params.path,
                height,
                width,
                params.rho,
                params.theta,
                params.line_threshold,
                params.min_line_length,
                params.min_line_gap,
            );
            // let img_with_lines_bytes = draw_lines();
            tx_lines.send(lines_mask).unwrap();

            // Denoising
            println!("Denoising...");
            let denoised_bytes = Self::clean_captcha(
                &inpainted.clone(),
                params.block,
                params.delta,
                params.erase_k,
                params.dilate_k,
                &bit_map,
                params.threshold,
            );
            tx_denoised.send(denoised_bytes.clone()).unwrap();

            println!("Done");
        });
    }

    fn load_texture_from_bytes(
        ui: &mut egui::Ui,
        bytes: Vec<u8>,
        texture: &mut Option<egui::TextureHandle>,
    ) {
        let image = image::load_from_memory(&bytes).unwrap().to_rgba8();

        let size = [image.width() as usize, image.height() as usize];

        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());

        Self::load_texture_from_color_image(ui, color_image, texture)
    }

    fn load_texture_from_color_image(
        ui: &mut egui::Ui,
        color_image: ColorImage,
        texture: &mut Option<egui::TextureHandle>,
    ) {
        *texture = Some(ui.load_texture("data", color_image, Default::default()));
    }
}

impl eframe::App for DenoiserApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Denoiser");

                ui.add_space(10.0);

                // ======= Original ========
                ui.label("Original");
                if let Ok(bytes) = self.rx_orig.try_recv() {
                    Self::load_texture_from_bytes(ui, bytes, &mut self.texture_orig);
                }

                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Some(texture) = &self.texture_orig {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                // ======= Density ========
                ui.label("Density map");
                if let Ok(bytes) = self.rx_density.try_recv() {
                    Self::load_texture_from_bytes(ui, bytes, &mut self.texture_denity);
                }

                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Some(texture) = &self.texture_denity {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                // ======= Lines ========
                ui.label("Lines detection");

                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Ok(color_image) = self.rx_lines.try_recv() {
                    Self::load_texture_from_color_image(ui, color_image, &mut self.texture_lines);
                }

                if let Some(texture) = &self.texture_lines {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                // ======= Denoised ========
                ui.label("Denoised");
                if let Ok(bytes) = self.rx_denoised.try_recv() {
                    Self::load_texture_from_bytes(ui, bytes, &mut self.texture_denoised);
                }
                // Image
                let size = egui::vec2(400.0, 100.0);

                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());

                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

                if let Some(texture) = &self.texture_denoised {
                    ui.put(rect, egui::Image::new(texture).fit_to_exact_size(size));
                } else {
                    ui.put(
                        rect,
                        egui::Label::new("No image").sense(egui::Sense::hover()),
                    );
                }

                let mut changed = false;
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    // 1. Constrain the width so the "center" is meaningful
                    ui.set_max_width(350.0);
                    egui::Grid::new("my_grid")
                        .num_columns(2)
                        .spacing([40.0, 4.0]) // [horizontal, vertical] spacing
                        .show(ui, |ui| {
                            ui.add(egui::Label::new("Block"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.block))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("RHO"));
                            if ui.add(egui::DragValue::new(&mut self.params.rho)).changed() {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Delta"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.delta))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("Theta"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.theta))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Erase K"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.erase_k))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("Line_threshold"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.line_threshold))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Dilate K"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.dilate_k))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("min_line_length"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.min_line_length))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();

                            ui.add(egui::Label::new("Threshold"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.threshold))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.add(egui::Label::new("min_line_gap"));
                            if ui
                                .add(egui::DragValue::new(&mut self.params.min_line_gap))
                                .changed()
                            {
                                self.denoise();
                            };

                            ui.end_row();
                        });
                });

                ui.add_space(20.0);
                // Button
                let btn = egui::Button::new(egui::RichText::new("Denoise").size(20.0));
                ui.vertical_centered(|ui| {
                    if ui.add(btn).clicked() {
                        self.denoise();
                    }
                });
            });
        });
    }
}

#[tokio::main]
async fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(540.0, 800.0)),
        ..Default::default()
    };

    let path = r"D:\Main3\captcha.png".to_string();

    let _ = eframe::run_native(
        "Denoiser",
        options,
        Box::new(|cc| Ok(Box::new(DenoiserApp::new(cc, path)))),
    );
}
