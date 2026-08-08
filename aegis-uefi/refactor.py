import os
import glob
import re

SRC_DIR = "/home/killboxincorporated/antigravity-aegis/src"
DEST_DIR = "/home/killboxincorporated/aegis-uefi/src"

files_to_copy = ["attention.rs", "inference.rs", "kvcache.rs", "model.rs", "ops.rs", "sampler.rs", "tokenizer.rs"]

for file in files_to_copy:
    with open(os.path.join(SRC_DIR, file), "r") as f:
        content = f.read()

    # Refactor std -> alloc/core
    content = content.replace("use std::collections::HashMap;", "use hashbrown::HashMap;")
    content = content.replace("std::string::String", "alloc::string::String")
    content = content.replace("std::vec::Vec", "alloc::vec::Vec")
    content = content.replace("std::sync::Arc", "alloc::sync::Arc")
    content = content.replace("std::boxed::Box", "alloc::boxed::Box")
    content = content.replace("use std::fs;", "")
    content = content.replace("use std::io;", "")
    content = content.replace("use std::io::Write;", "")
    content = content.replace("std::io::Error", "core::fmt::Error")
    
    # In rust, String, Vec, etc are exported in alloc
    content = "#![no_std]\nextern crate alloc;\nuse alloc::{vec::Vec, string::String, vec, format, string::ToString, sync::Arc, boxed::Box};\n" + content

    # Write to destination
    with open(os.path.join(DEST_DIR, file), "w") as f:
        f.write(content)

print("Files copied and initially refactored.")
