//! This crate implements Rerun's code generation tools.
//!
//! These tools translate language-agnostic IDL definitions (flatbuffers) into code.
//!
//! They are invoked by `pixi run codegen`.
//!
//!
//! ### Organization
//!
//! The code generation process happens in 4 phases.
//!
//! #### 1. Generate binary reflection data from flatbuffers definitions.
//!
//! All this does is invoke the flatbuffers compiler (`flatc`) with the right flags in order to
//! generate the binary dumps.
//!
//! Look for `compile_binary_schemas` in the code.
//!
//! #### 2. Run the semantic pass.
//!
//! The semantic pass transforms the low-level raw reflection data generated by the first phase
//! into higher level objects that are much easier to inspect/manipulate and overall friendlier
//! to work with.
//!
//! Look for `objects.rs`.
//!
//! #### 3. Fill the Arrow registry.
//!
//! The Arrow registry keeps track of all type definitions and maps them to Arrow datatypes.
//!
//! Look for `type_registry.rs`.
//!
//! #### 4. Run the actual codegen pass for a given language.
//!
//! We currently have two different codegen passes implemented at the moment: Python & Rust.
//!
//! Codegen passes use the semantic objects from phase two and the registry from phase three
//! in order to generate user-facing code for Rerun's SDKs.
//!
//! These passes are intentionally implemented using a very low-tech no-frills approach (stitch
//! strings together, make liberal use of `unimplemented`, etc) that keep them flexible in the
//! face of ever changing needs in the generated code.
//!
//! Look for `codegen/python.rs` and `codegen/rust.rs`.
//!
//!
//! ### Error handling
//!
//! Keep in mind: this is all _build-time_ code that will never see the light of runtime.
//! There is therefore no need for fancy error handling in this crate: all errors are fatal to the
//! build anyway.
//!
//! Make sure to crash as soon as possible when something goes wrong and to attach all the
//! appropriate/available context using `anyhow`'s `with_context` (e.g. always include the
//! fully-qualified name of the faulty type/field) and you're good to go.
//!
//!
//! ### Testing
//!
//! Same comment as with error handling: this code becomes irrelevant at runtime, and so testing it
//! brings very little value.
//!
//! Make sure to test the behavior of its output though: `re_types`!
//!
//!
//! ### Understanding the subtleties of affixes
//!
//! So-called "affixes" are effects applied to objects defined with the Rerun IDL and that affect
//! the way these objects behave and interoperate with each other (so, yes, monads. shhh.).
//!
//! There are 3 distinct and very common affixes used when working with Rerun's IDL: transparency,
//! nullability and plurality.
//!
//! Broadly, we can describe these affixes as follows:
//! - Transparency allows for bypassing a single layer of typing (e.g. to "extract" a field out of
//!   a struct).
//! - Nullability specifies whether a piece of data is allowed to be left unspecified at runtime.
//! - Plurality specifies whether a piece of data is actually a collection of that same type.
//!
//! We say "broadly" here because the way these affixes ultimately affect objects in practice will
//! actually depend on the kind of object that they are applied to, of which there are 3: archetypes,
//! components and datatypes.
//!
//! Not only that, but objects defined in Rerun's IDL are materialized into 3 distinct environments:
//! IDL definitions, Arrow datatypes and native code (e.g. Rust & Python).
//!
//! These environment have vastly different characteristics, quirks, pitfalls and limitations,
//! which once again lead to these affixes having different, sometimes surprising behavior
//! depending on the environment we're interested in.
//! Also keep in mind that Flatbuffers and native code are generally designed around arrays of
//! structures, while Arrow is all about structures of arrays!
//!
//! All in all, these interactions between affixes, object kinds and environments lead to a
//! combinatorial explosion of edge cases that can be very confusing when it comes to (de)serialization
//! code, and even API design.
//!
//! When in doubt, check out the `rerun.testing.archetypes.AffixFuzzer` IDL definitions, generated code and
//! test suites for definitive answers.

// TODO(#2365): support for external IDL definitions

// ---

// TODO(#6330): remove unwrap()
#![allow(clippy::unwrap_used)]

// NOTE: Official generated code from flatbuffers; ignore _everything_.

#[allow(
    warnings,
    unused,
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    dead_code,
    unused_imports,
    explicit_outlives_requirements,
    clippy::all
)]
mod reflection;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::Context as _;
use codegen::FbsCodeGenerator;
use format::FbsCodeFormatter;
use re_build_tools::{
    compute_crate_hash, compute_dir_filtered_hash, compute_dir_hash, compute_strings_hash,
};

use crate::format::NoopCodeFormatter;

pub use self::reflection::reflection::{
    BaseType as FbsBaseType, Enum as FbsEnum, EnumVal as FbsEnumVal, Field as FbsField,
    KeyValue as FbsKeyValue, Object as FbsObject, Schema as FbsSchema, Type as FbsType,
    root_as_schema,
};

// NOTE: This crate isn't only okay with `unimplemented`, it actively encourages it.

#[allow(clippy::unimplemented)]
mod codegen;
#[allow(clippy::unimplemented)]
mod format;
#[allow(clippy::unimplemented)]
mod objects;
#[allow(clippy::unimplemented)]
mod type_registry;

pub mod data_type;
mod docs;

pub mod report;

/// In-memory generated files.
///
/// Generated by the codegen pass, modified in-place by the post-processing passes (formatting
/// etc), and finally written to disk by the I/O pass.
pub type GeneratedFiles = std::collections::BTreeMap<camino::Utf8PathBuf, String>;

pub use self::{
    codegen::{
        CodeGenerator, CppCodeGenerator, DocsCodeGenerator, PythonCodeGenerator, RustCodeGenerator,
        SnippetsRefCodeGenerator,
    },
    docs::Docs,
    format::{CodeFormatter, CppCodeFormatter, PythonCodeFormatter, RustCodeFormatter},
    objects::{
        Attributes, ElementType, Object, ObjectClass, ObjectField, ObjectKind, Objects, Type,
    },
    report::{Report, Reporter},
    type_registry::TypeRegistry,
};

// --- Attributes ---

pub const ATTR_DEFAULT: &str = "default";
pub const ATTR_NULLABLE: &str = "nullable";
pub const ATTR_ORDER: &str = "order";
pub const ATTR_TRANSPARENT: &str = "transparent";

pub const ATTR_ARROW_TRANSPARENT: &str = "attr.arrow.transparent";
pub const ATTR_ARROW_SPARSE_UNION: &str = "attr.arrow.sparse_union";

pub const ATTR_RERUN_COMPONENT_OPTIONAL: &str = "attr.rerun.component_optional";
pub const ATTR_RERUN_COMPONENT_RECOMMENDED: &str = "attr.rerun.component_recommended";
pub const ATTR_RERUN_COMPONENT_REQUIRED: &str = "attr.rerun.component_required";
pub const ATTR_RERUN_LOG_MISSING_AS_EMPTY: &str = "attr.rerun.log_missing_as_empty";
pub const ATTR_RERUN_OVERRIDE_TYPE: &str = "attr.rerun.override_type";
pub const ATTR_RERUN_SCOPE: &str = "attr.rerun.scope";
pub const ATTR_RERUN_VIEW_IDENTIFIER: &str = "attr.rerun.view_identifier";
pub const ATTR_RERUN_STATE: &str = "attr.rerun.state";
pub const ATTR_RERUN_DEPRECATED_SINCE: &str = "attr.rerun.deprecated_since";
pub const ATTR_RERUN_DEPRECATED_NOTICE: &str = "attr.rerun.deprecated_notice";

pub const ATTR_PYTHON_ALIASES: &str = "attr.python.aliases";
pub const ATTR_PYTHON_ARRAY_ALIASES: &str = "attr.python.array_aliases";

pub const ATTR_CPP_NO_DEFAULT_CTOR: &str = "attr.cpp.no_default_ctor";
pub const ATTR_CPP_NO_FIELD_CTORS: &str = "attr.cpp.no_field_ctors";
pub const ATTR_CPP_RENAME_FIELD: &str = "attr.cpp.rename_field";

pub const ATTR_RUST_CUSTOM_CLAUSE: &str = "attr.rust.custom_clause";
pub const ATTR_RUST_DERIVE: &str = "attr.rust.derive";
pub const ATTR_RUST_DERIVE_ONLY: &str = "attr.rust.derive_only";
pub const ATTR_RUST_NEW_PUB_CRATE: &str = "attr.rust.new_pub_crate";
pub const ATTR_RUST_OVERRIDE_CRATE: &str = "attr.rust.override_crate";
pub const ATTR_RUST_REPR: &str = "attr.rust.repr";
pub const ATTR_RUST_TUPLE_STRUCT: &str = "attr.rust.tuple_struct";

pub const ATTR_DOCS_UNRELEASED: &str = "attr.docs.unreleased";

pub const ATTR_DOCS_CATEGORY: &str = "attr.docs.category";
pub const ATTR_DOCS_VIEW_TYPES: &str = "attr.docs.view_types";

// --- Entrypoints ---

use camino::{Utf8Path, Utf8PathBuf};

/// Compiles binary reflection dumps from flatbuffers definitions.
///
/// Requires `flatc` available in $PATH.
///
/// Panics on error.
///
/// - `include_dir_path`: path to the root directory of the fbs definition tree.
/// - `output_dir_path`: output directory, where the binary schemas will be stored.
/// - `entrypoint_path`: path to the root file of the fbs definition tree.
pub fn compile_binary_schemas(
    include_dir_path: impl AsRef<Utf8Path>,
    output_dir_path: impl AsRef<Utf8Path>,
    entrypoint_path: impl AsRef<Utf8Path>,
) {
    let include_dir_path = include_dir_path.as_ref().as_str();
    let output_dir_path = output_dir_path.as_ref().as_str();
    let entrypoint_path = entrypoint_path.as_ref().as_str();

    use xshell::{Shell, cmd};
    let sh = Shell::new().unwrap();
    cmd!(
        sh,
        "flatc -I {include_dir_path}
            -o {output_dir_path}
            -b --bfbs-comments --schema
            {entrypoint_path}"
    )
    .run()
    .unwrap();
}

/// Handles the first 3 language-agnostic passes of the codegen pipeline:
/// 1. Generate binary reflection dumps for our definitions.
/// 2. Run the semantic pass
/// 3. Compute the Arrow registry
///
/// Panics on error.
///
/// - `include_dir_path`: path to the root directory of the fbs definition tree.
/// - `entrypoint_path`: path to the root file of the fbs definition tree.
pub fn generate_lang_agnostic(
    reporter: &Reporter,
    include_dir_path: impl AsRef<Utf8Path>,
    entrypoint_path: impl AsRef<Utf8Path>,
) -> (Objects, TypeRegistry) {
    re_tracing::profile_function!();

    use xshell::Shell;

    let sh = Shell::new().unwrap();
    let tmp = sh.create_temp_dir().unwrap();
    let tmp_path = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();

    let entrypoint_path = entrypoint_path.as_ref();
    let entrypoint_filename = entrypoint_path.file_name().unwrap();

    let include_dir_path = include_dir_path.as_ref();
    let include_dir_path = include_dir_path
        .canonicalize_utf8()
        .with_context(|| format!("failed to canonicalize include path: {include_dir_path:?}"))
        .unwrap();

    // generate bfbs definitions
    compile_binary_schemas(&include_dir_path, &tmp_path, entrypoint_path);

    let mut binary_entrypoint_path = Utf8PathBuf::from(entrypoint_filename);
    binary_entrypoint_path.set_extension("bfbs");

    // semantic pass: high level objects from low-level reflection data
    let mut objects = Objects::from_buf(
        reporter,
        include_dir_path,
        sh.read_binary_file(tmp_path.join(binary_entrypoint_path))
            .unwrap()
            .as_slice(),
    );

    // create and fill out arrow registry
    let mut type_registry = TypeRegistry::default();
    for obj in objects.objects.values_mut() {
        type_registry.register(obj);
    }

    (objects, type_registry)
}

/// Generates .gitattributes files that mark up all generated files as generated.
fn generate_gitattributes_for_generated_files(files_to_write: &mut GeneratedFiles) {
    re_tracing::profile_function!();

    const FILENAME: &str = ".gitattributes";

    let mut filepaths_per_folder = BTreeMap::default();

    for filepath in files_to_write.keys() {
        let dirpath = filepath.parent().unwrap();
        let files: &mut Vec<_> = filepaths_per_folder.entry(dirpath.to_owned()).or_default();
        files.push(filepath.clone());
    }

    for (dirpath, files) in filepaths_per_folder {
        let gitattributes_path = dirpath.join(FILENAME);

        let generated_files = std::iter::once(FILENAME.to_owned()) // The attributes itself is generated!
            .chain(files.iter().map(|filepath| {
                format_path(
                    filepath
                        .strip_prefix(&dirpath)
                        .with_context(|| {
                            format!("Failed to make {filepath} relative to {dirpath}.")
                        })
                        .unwrap(),
                )
            }))
            .map(|s| format!("{s} linguist-generated=true"))
            .collect::<Vec<_>>();

        let content = format!(
            "# DO NOT EDIT! This file is generated by {}\n\n{}\n",
            format_path(file!()),
            generated_files.join("\n")
        );

        files_to_write.insert(gitattributes_path, content);
    }
}

/// This will automatically emit a `rerun-if-changed` clause for all the files that were hashed.
pub fn compute_re_types_builder_hash() -> String {
    re_tracing::profile_function!();

    compute_crate_hash("re_types_builder")
}

pub struct SourceLocations<'a> {
    pub definitions_dir: &'a str,
    pub snippets_dir: &'a str,
    pub python_output_dir: &'a str,
    pub cpp_output_dir: &'a str,
}

/// Also triggers a re-build if anything that affects the hash changes.
pub fn compute_re_types_hash(locations: &SourceLocations<'_>) -> String {
    re_tracing::profile_function!();

    // NOTE: We need to hash both the flatbuffers definitions as well as the source code of the
    // code generator itself!
    let re_types_builder_hash = compute_re_types_builder_hash();
    let definitions_hash = compute_dir_hash(locations.definitions_dir, Some(&["fbs"]));
    let snippets_hash = compute_dir_hash(locations.snippets_dir, Some(&["rs", "py", "cpp"]));
    let snippets_index_hash = PathBuf::from(locations.snippets_dir)
        .parent()
        .map_or(String::new(), |dir| {
            compute_dir_filtered_hash(dir, |path| path.to_str().unwrap().ends_with("INDEX.md"))
        });
    let python_extensions_hash = compute_dir_filtered_hash(locations.python_output_dir, |path| {
        path.to_str().unwrap().ends_with("_ext.py")
    });
    let cpp_extensions_hash = compute_dir_filtered_hash(locations.cpp_output_dir, |path| {
        path.to_str().unwrap().ends_with("_ext.cpp")
    });

    let new_hash = compute_strings_hash(&[
        &re_types_builder_hash,
        &definitions_hash,
        &snippets_hash,
        &snippets_index_hash,
        &python_extensions_hash,
        &cpp_extensions_hash,
    ]);

    re_log::debug!("re_types_builder_hash: {re_types_builder_hash:?}");
    re_log::debug!("definitions_hash: {definitions_hash:?}");
    re_log::debug!("snippets_hash: {snippets_hash:?}");
    re_log::debug!("snippets_index_hash: {snippets_index_hash:?}");
    re_log::debug!("python_extensions_hash: {python_extensions_hash:?}");
    re_log::debug!("cpp_extensions_hash: {cpp_extensions_hash:?}");
    re_log::debug!("new_hash: {new_hash:?}");

    new_hash
}

/// Generates, formats and optionally writes code.
///
/// If `check` is true, this will run a comparison check instead of writing files to disk.
///
/// Panics on error.
fn generate_code(
    reporter: &Reporter,
    objects: &Objects,
    type_registry: &TypeRegistry,
    generator: &mut dyn CodeGenerator,
    formatter: &mut dyn CodeFormatter,
    orphan_paths_opt_out: &BTreeSet<Utf8PathBuf>,
    check: bool,
) {
    use rayon::prelude::*;

    // Generate in-memory code files:
    let mut files = generator.generate(reporter, objects, type_registry);

    for (filepath, contents) in &files {
        if !contents.contains("DO NOT EDIT") {
            reporter.error_file(
                filepath,
                "Generated file missing DO NOT EDIT warning. Use the `autogen_warning!` macro.",
            );
        }
    }

    // Generate in-memory gitattribute files:
    generate_gitattributes_for_generated_files(&mut files);

    // Format in-memory files:
    formatter.format(reporter, &mut files);

    if check {
        files.par_iter().for_each(|(filepath, contents)| {
            let cur_contents = std::fs::read_to_string(filepath).unwrap();
            if *contents != cur_contents {
                reporter.error(filepath.as_str(), "", "out of sync");
            }
        });
        return;
    }

    // Write all files to filesystem:
    {
        re_tracing::profile_wait!("write_files");

        files.par_iter().for_each(|(filepath, contents)| {
            crate::codegen::common::write_file(filepath, contents);
        });
    }

    // Remove orphaned files:
    for path in orphan_paths_opt_out {
        files.retain(|filepath, _| filepath.parent() != Some(path));
    }
    crate::codegen::common::remove_orphaned_files(reporter, &files);
}

/// Generates C++ code.
///
/// If `check` is true, this will run a comparison check instead of writing files to disk.
///
/// Panics on error.
///
/// - `output_path`: path to the root of the output.
pub fn generate_cpp_code(
    reporter: &Reporter,
    output_path: impl AsRef<Utf8Path>,
    objects: &Objects,
    type_registry: &TypeRegistry,
    check: bool,
) {
    re_tracing::profile_function!();

    let mut generator = CppCodeGenerator::new(output_path.as_ref());
    let mut formatter = CppCodeFormatter;

    // NOTE: In rerun_cpp we have a directory where we share generated code with handwritten code.
    // Make sure to filter out that directory, or else we will end up removing those handwritten
    // files.
    let orphan_path_opt_out = output_path.as_ref().join("src/rerun");

    generate_code(
        reporter,
        objects,
        type_registry,
        &mut generator,
        &mut formatter,
        &std::iter::once(orphan_path_opt_out).collect(),
        check,
    );
}

/// Generates Rust code.
///
/// If `check` is true, this will run a comparison check instead of writing files to disk.
///
/// Panics on error.
pub fn generate_rust_code(
    reporter: &Reporter,
    workspace_path: impl Into<Utf8PathBuf>,
    objects: &Objects,
    type_registry: &TypeRegistry,
    check: bool,
) {
    re_tracing::profile_function!();

    let mut generator = RustCodeGenerator::new(workspace_path);
    let mut formatter = RustCodeFormatter;

    generate_code(
        reporter,
        objects,
        type_registry,
        &mut generator,
        &mut formatter,
        &Default::default(),
        check,
    );
}

/// Generates Python code.
///
/// If `check` is true, this will run a comparison check instead of writing files to disk.
///
/// Panics on error.
///
/// - `output_pkg_path`: path to the root of the output package.
pub fn generate_python_code(
    reporter: &Reporter,
    output_pkg_path: impl AsRef<Utf8Path>,
    testing_output_pkg_path: impl AsRef<Utf8Path>,
    objects: &Objects,
    type_registry: &TypeRegistry,
    check: bool,
) {
    re_tracing::profile_function!();

    let mut generator =
        PythonCodeGenerator::new(output_pkg_path.as_ref(), testing_output_pkg_path.as_ref());
    let mut formatter =
        PythonCodeFormatter::new(output_pkg_path.as_ref(), testing_output_pkg_path.as_ref());

    generate_code(
        reporter,
        objects,
        type_registry,
        &mut generator,
        &mut formatter,
        &Default::default(),
        check,
    );
}

pub fn generate_docs(
    reporter: &Reporter,
    output_docs_dir: impl AsRef<Utf8Path>,
    objects: &Objects,
    type_registry: &TypeRegistry,
    check: bool,
) {
    re_tracing::profile_function!();

    re_log::info!("Generating docs to {}", output_docs_dir.as_ref());

    let mut generator = DocsCodeGenerator::new(output_docs_dir.as_ref());
    let mut formatter = NoopCodeFormatter;

    generate_code(
        reporter,
        objects,
        type_registry,
        &mut generator,
        &mut formatter,
        &Default::default(),
        check,
    );
}

pub fn generate_snippets_ref(
    reporter: &Reporter,
    output_snippets_ref_dir: impl AsRef<Utf8Path>,
    objects: &Objects,
    type_registry: &TypeRegistry,
    check: bool,
) {
    re_tracing::profile_function!();

    re_log::info!(
        "Generating snippets_ref to {}",
        output_snippets_ref_dir.as_ref()
    );

    let mut generator = SnippetsRefCodeGenerator::new(output_snippets_ref_dir.as_ref());
    let mut formatter = NoopCodeFormatter;

    // NOTE: We generate in an existing folder, don't touch anything else.
    let orphan_path_opt_out = output_snippets_ref_dir.as_ref().to_owned();

    generate_code(
        reporter,
        objects,
        type_registry,
        &mut generator,
        &mut formatter,
        &std::iter::once(orphan_path_opt_out).collect(),
        check,
    );
}

/// Generate flatbuffers definition files.
///
/// This should run as the first step in the codegen pipeline as it influences all others.
pub fn generate_fbs(reporter: &Reporter, definition_dir: impl AsRef<Utf8Path>, check: bool) {
    re_tracing::profile_function!();

    let mut generator = FbsCodeGenerator::new(definition_dir.as_ref());
    let mut formatter = FbsCodeFormatter;

    // We don't have arrow registry & objects yet!
    let objects = Objects::default();
    let type_registry = TypeRegistry::default();

    let orphan_path_opt_outs = [
        definition_dir.as_ref().to_path_buf(),
        definition_dir.as_ref().join("rerun"),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();

    generate_code(
        reporter,
        &objects,
        &type_registry,
        &mut generator,
        &mut formatter,
        &orphan_path_opt_outs,
        check,
    );
}

pub(crate) fn rerun_workspace_path() -> camino::Utf8PathBuf {
    let workspace_root = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let manifest_dir = camino::Utf8PathBuf::from(manifest_dir)
            .canonicalize_utf8()
            .unwrap();
        manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .unwrap()
            .to_path_buf()
    } else {
        let file_path = camino::Utf8PathBuf::from(file!())
            .canonicalize_utf8()
            .unwrap();
        file_path
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .unwrap()
            .to_path_buf()
    };

    assert!(
        workspace_root.exists(),
        "Failed to find workspace root, expected it at {workspace_root:?}"
    );

    // Check for something that only exists in root:
    assert!(
        workspace_root.join("CODE_OF_CONDUCT.md").exists(),
        "Failed to find workspace root, expected it at {workspace_root:?}"
    );

    workspace_root.canonicalize_utf8().unwrap()
}

/// Format the path with forward slashes, even on Windows.
pub(crate) fn format_path(path: impl AsRef<Utf8Path>) -> String {
    path.as_ref().as_str().replace('\\', "/")
}
