extern crate nalgebra as na;

use std::fs::File;

use eframe::{
    egui::{panel::Side, CentralPanel, Context, Image, SidePanel},
    epaint::{vec2, AlphaImage, ColorImage, ImageData, TextureHandle, TextureId},
    epi::{App, Frame, Storage},
    NativeOptions,
};
use image::{load_from_memory, GrayImage};
use network::{ConvolutionalLayer, Network};

mod mnist;
mod network;

fn main() {
    let mut network = Network::new(na::Vector2::new(28, 28));
    network.add_layer(Box::new(ConvolutionalLayer::new(na::Vector2::new(5, 5))));

    let images = mnist::parse_mnist_images(
        &mut File::open("assets/samples/mnist/train-images-idx3-ubyte").unwrap(),
    );
    let labels = mnist::parse_mnist_labels(
        &mut File::open("assets/samples/mnist/train-labels-idx1-ubyte").unwrap(),
    );
    let network_app = NetworkApp::new(network, images, labels);

    eframe::run_native(Box::new(network_app), NativeOptions::default());
}

pub struct NetworkApp {
    network: Network,

    node_size: f32,
    node_spacing: f32,
    layer_spacing: f32,

    images: Vec<GrayImage>,
    labels: Vec<u8>,

    curr_image_index: usize,

    input_texture: Option<TextureHandle>,
    layer_textures: Vec<TextureHandle>,
}

impl NetworkApp {
    pub fn new(network: Network, images: Vec<GrayImage>, labels: Vec<u8>) -> Self {
        Self {
            network,

            node_size: 10.0,
            node_spacing: 15.0,
            layer_spacing: 100.0,

            images,
            labels,

            curr_image_index: 0,

            input_texture: None,
            layer_textures: Vec::new(),
        }
    }
}

impl App for NetworkApp {
    fn name(&self) -> &str {
        "Network Visualization"
    }

    fn setup(&mut self, ctx: &Context, _frame: &Frame, _storage: Option<&dyn Storage>) {
        let curr_image = &self.images[self.curr_image_index];
        self.input_texture = Some(load_image_to_texture(ctx, curr_image));

        let output = self.network.evaluate(
            &curr_image
                .iter()
                .cloned()
                .map(|pixel| (pixel as f32) / 255.0)
                .collect::<Vec<_>>(),
        );

        let output_size = self.network.output_size();
        
        self.layer_textures.push(load_image_to_texture(
            ctx,
            &GrayImage::from_vec(
                output_size.x.try_into().unwrap(),
                output_size.y.try_into().unwrap(),
                output.iter().map(|pixel| (pixel * 255.0) as u8).collect::<Vec<_>>(),
            ).unwrap(),
        ));
        // for layer in self.network.layers().iter() {
        // }
    }

    fn update(&mut self, ctx: &Context, frame: &Frame) {
        let curr_image = &self.images[self.curr_image_index];
        self.input_texture = Some(load_image_to_texture(ctx, curr_image));

        let output = self.network.evaluate(
            &curr_image
                .iter()
                .cloned()
                .map(|pixel| (pixel as f32) / 255.0)
                .collect::<Vec<_>>(),
        );

        let output_size = self.network.output_size();
        
        self.layer_textures = vec![(load_image_to_texture(
            ctx,
            &GrayImage::from_vec(
                output_size.x.try_into().unwrap(),
                output_size.y.try_into().unwrap(),
                output.iter().map(|pixel| (pixel * 255.0) as u8).collect::<Vec<_>>(),
            ).unwrap(),
        ))];
        
        CentralPanel::default().show(ctx, |ui| {
            if let Some(input_texture) = &self.input_texture {
                ui.add(Image::new(input_texture, vec2(56.0, 56.0)));
            }

            for layer_texture in self.layer_textures.iter() {
                ui.add(Image::new(layer_texture, vec2(56.0, 56.0)));
            }
        });
    }
}

fn load_image_to_texture(ctx: &Context, image: &GrayImage) -> TextureHandle {
    ctx.load_texture(
        "input",
        ColorImage::from_rgba_unmultiplied(
            [
                image.width().try_into().unwrap(),
                image.height().try_into().unwrap(),
            ],
            &image
                .iter()
                .copied()
                .flat_map(|pixel| [255, 255, 255, pixel])
                .collect::<Vec<_>>(),
        ),
    )
}
