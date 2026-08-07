import os

# Fix ops.rs
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "r") as f:
    content = f.read()

old_matvec = """pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_i8: &[i8]) {
    // Branchless formulation for Ternary weights (-1, 0, 1)
    let in_len = input.len();
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        let w_row = &weights_i8[i * in_len..(i + 1) * in_len];
        for (j, &in_val) in input.iter().enumerate() {
            let w = w_row[j] as f32;
            sum += in_val * w;
        }
        *out = sum;
    }
}"""

new_matvec = """pub fn ternary_matvec(output: &mut [f32], input: &[f32], weights_u8: &[u8], dim_out: usize, dim_in: usize) {
    for i in 0..dim_out {
        let mut sum = 0.0;
        let w_row = &weights_u8[i * dim_in..(i + 1) * dim_in];
        for j in 0..dim_in {
            let w = w_row[j] as i8 as f32;
            sum += input[j] * w;
        }
        output[i] = sum;
    }
}"""
content = content.replace(old_matvec, new_matvec)
with open("/home/killboxincorporated/aegis-uefi/src/ops.rs", "w") as f:
    f.write(content)

# Fix main.rs FileInfo
with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "r") as f:
    content = f.read()

content = content.replace("use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType, FileInfo};", "use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};")
content = content.replace("let info: &FileInfo = file.get_info(&mut info_buf).ok()?;", "let info = file.get_info::<uefi::proto::media::file::FileInfo>(&mut info_buf).ok()?;")
with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "w") as f:
    f.write(content)
