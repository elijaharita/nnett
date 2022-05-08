use std::{fs::File, io::Read};

use image::{ImageBuffer, GrayImage};

pub fn parse_mnist_images(file: &mut File) -> Vec<GrayImage> {
    let magic_number = read_u32(file);
    if magic_number != 2051 {
        panic!("wrong magic number: {}", magic_number)
    }

    let image_count = read_u32(file);
    let image_width = read_u32(file);
    let image_height = read_u32(file);

    (0..image_count).map(|_| {
        let mut pixels = vec![0u8; (image_width * image_height) as usize];
        file.read_exact(&mut pixels).unwrap();
        ImageBuffer::from_vec(image_width, image_height, pixels).unwrap()
    }).collect()
}

pub fn parse_mnist_labels(file: &mut File) -> Vec<u8> {
    let magic_number = read_u32(file);
    if magic_number != 2049 {
        panic!("wrong magic number: {}", magic_number)
    }
    
    let item_count = read_u32(file);

    (0..item_count).map(|_| read_u8(file)).collect()
}

fn read_u32(file: &mut File) -> u32 {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes).unwrap();
    u32::from_be_bytes(bytes)
}

fn read_u8(file: &mut File) -> u8 {
    let mut bytes = [0u8; 1];
    file.read_exact(&mut bytes).unwrap();
    bytes[0]
}