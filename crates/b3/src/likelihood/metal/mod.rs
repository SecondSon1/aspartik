#![expect(clippy::undocumented_unsafe_blocks)]

use anyhow::{Context, Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
	MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
	MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
	MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
	MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};
use parking_lot::MutexGuard;

use core::ffi::c_void;
use core::ptr::NonNull;

use super::Calculator;
use crate::{Transitions, parameters::Tree};

type Transition = [f64; 16];

type GpuRow = [f32; 4];
type GpuTransition = [f32; 16];

const METAL_SRC: &str = include_str!("kernels.metal");

/// Allocate a zeroed shared-mode Metal buffer holding `count` values of type `T`.
fn alloc_buf<T>(
	device: &ProtocolObject<dyn MTLDevice>,
	count: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
	let bytes = count * size_of::<T>();
	let buf = device
		.newBufferWithLength_options(bytes, MTLResourceOptions::empty())
		.ok_or_else(|| anyhow!("Metal: alloc {bytes} bytes failed"))?;

	// Zero-initialise (contents() is shared memory on Apple Silicon)
	unsafe {
		buf.contents().as_ptr().cast::<u8>().write_bytes(0, bytes);
	}
	Ok(buf)
}

/// Create a shared-mode Metal buffer pre-populated with `data`.
fn upload_buf<T: bytemuck::Pod>(
	device: &ProtocolObject<dyn MTLDevice>,
	data: &[T],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
	let bytes = data.len() * size_of::<T>();
	let buf_ptr = NonNull::new(data.as_ptr() as *mut c_void)
		.ok_or_else(|| anyhow!("Metal: null data pointer"))?;
	// SAFETY: `data` outlives the call; Metal copies the bytes before
	// returning.
	let buf = unsafe {
		device.newBufferWithBytes_length_options(
			buf_ptr,
			bytes,
			MTLResourceOptions::empty(),
		)
	}
	.ok_or_else(|| anyhow!("Metal: upload {bytes} bytes failed"))?;
	Ok(buf)
}

/// Write a slice into a (shared-mode) Metal buffer.
///
/// # Safety
/// The buffer must be large enough to hold `data`.
unsafe fn write_buf<T: bytemuck::Pod>(
	buf: &ProtocolObject<dyn MTLBuffer>,
	data: &[T],
) {
	let ptr = buf.contents().as_ptr().cast::<T>();
	unsafe {
		ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
	}
}

/// Read `count` values of type `T` from a (shared-mode) Metal buffer.
///
/// # Safety
/// The buffer must hold at least `count` values of type `T`, and no GPU
/// work writing to it may be in flight.
unsafe fn read_slice<T: bytemuck::Pod + Copy>(
	buf: &ProtocolObject<dyn MTLBuffer>,
	count: usize,
) -> Vec<T> {
	let ptr = buf.contents().as_ptr().cast::<T>();
	unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
}

pub struct MetalLikelihood {
	_device: Retained<ProtocolObject<dyn MTLDevice>>,
	queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

	propose_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
	update_leaves_pipeline:
		Retained<ProtocolObject<dyn MTLComputePipelineState>>,
	update_likelihoods_pipeline:
		Retained<ProtocolObject<dyn MTLComputePipelineState>>,
	copy_projections_pipeline:
		Retained<ProtocolObject<dyn MTLComputePipelineState>>,

	/// Leaf bitmask nucleotides  — num_leaves × num_patterns u8
	leaves: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Per-edge partial likelihoods — num_nodes × num_patterns × 4 f32
	projections: Retained<ProtocolObject<dyn MTLBuffer>>,
	projections_backup: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Per-pattern root log-likelihoods — num_patterns f32
	likelihoods: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Flat children list for internal nodes — num_internals × 2 u32
	children: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Per-updated-edge transition matrices — num_updated_nodes × 4 f32x4 (f32)
	transitions: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Updated node indices for the current step — num_nodes u32
	nodes: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Per-(node,pattern) scaling flag — num_nodes × num_patterns u8
	scales: Retained<ProtocolObject<dyn MTLBuffer>>,
	scales_backup: Retained<ProtocolObject<dyn MTLBuffer>>,

	/// Per-pattern accumulated scaling counts — num_patterns u32
	scale_sums: Retained<ProtocolObject<dyn MTLBuffer>>,
	scale_sums_backup: Retained<ProtocolObject<dyn MTLBuffer>>,

	pattern_weights: Vec<u32>,
	num_patterns: u32,
	num_updated_nodes: u32,
}

// SAFETY: Retained is atomically counted smart pointer
unsafe impl Send for MetalLikelihood {}

impl Calculator<4, f64> for MetalLikelihood {
	fn likelihood(
		&mut self,
		mut tree: MutexGuard<Tree>,
		transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		let (nodes, children, leaves_end) = tree.propagation_lists();
		let tms = transitions.matrices(&nodes[..nodes.len() - 1]);

		self.num_updated_nodes = nodes.len() as u32 - 1;

		let root_children = *children.last().unwrap();
		let nodes_u32: Vec<u32> =
			nodes.iter().map(|&n| n as u32).collect();
		let children_u32: Vec<u32> = children
			.iter()
			.flat_map(|&[l, r]| [l as u32, r as u32])
			.collect();
		let tms_f32: Vec<GpuTransition> = {
			let tms_flat: &[Transition] =
				bytemuck::cast_slice(&tms);
			tms_flat.iter().map(|t| t.map(|v| v as f32)).collect()
		};

		// SAFETY: buffers were allocated with the correct sizes.
		unsafe {
			write_buf(&self.nodes, &nodes_u32);
			write_buf(&self.children, &children_u32);
			write_buf(&self.transitions, &tms_f32);
		}

		let mut leaves_end = leaves_end as u32;
		let internals_start = leaves_end;

		let cmd = self.command_buffer()?;

		if leaves_end > 10 {
			self.encode_update_leaves(&cmd, leaves_end)?;
			leaves_end = 0;
		}
		self.encode_update_all(&cmd, leaves_end, internals_start)?;

		let root = *nodes_u32.last().unwrap();
		let freq_f32: [f32; 4] =
			transitions.frequencies().map(|v| v as f32);
		self.encode_update_likelihoods(
			&cmd,
			root,
			(root_children[0] as u32, root_children[1] as u32),
			freq_f32,
		)?;

		drop(tree);

		cmd.commit();
		cmd.waitUntilCompleted();
		Self::check_completion(&cmd)?;

		// SAFETY: GPU work is complete, buffers are now CPU-readable.
		// Likelihoods are stored as f32 on the GPU; promote to f64 here.
		let likelihoods_f32: Vec<f32> = unsafe {
			read_slice(
				&self.likelihoods,
				self.num_patterns as usize,
			)
		};
		let scale_sums: Vec<u32> = unsafe {
			read_slice(&self.scale_sums, self.num_patterns as usize)
		};

		let total: f64 = likelihoods_f32
			.iter()
			.zip(&scale_sums)
			.zip(&self.pattern_weights)
			.map(|((l, &scale), &weight)| {
				(f64::from(*l) - f64::from(scale))
					* f64::from(weight)
			})
			.sum();

		Ok(total)
	}

	fn accept(&mut self) -> Result<()> {
		let cmd = self.command_buffer()?;
		self.encode_blit_scale_sums(&cmd, true)?;
		self.encode_copy_projections(&cmd, true)?;
		cmd.commit();
		cmd.waitUntilCompleted();
		Self::check_completion(&cmd)?;
		self.num_updated_nodes = 0;
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		let cmd = self.command_buffer()?;
		self.encode_blit_scale_sums(&cmd, false)?;
		self.encode_copy_projections(&cmd, false)?;
		cmd.commit();
		cmd.waitUntilCompleted();
		Self::check_completion(&cmd)?;
		self.num_updated_nodes = 0;
		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns as usize
	}
}

impl MetalLikelihood {
	/// Encode the leaf-projection update kernel onto `cmd`.
	fn encode_update_leaves(
		&self,
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
		leaves_end: u32,
	) -> Result<()> {
		let enc = self.compute_encoder(cmd)?;

		let block_size: usize = 16;
		let num_pattern_blocks =
			(self.num_patterns as usize).div_ceil(block_size);

		enc.setComputePipelineState(&self.update_leaves_pipeline);
		unsafe {
			enc.setBuffer_offset_atIndex(Some(&self.leaves), 0, 0);
			enc.setBuffer_offset_atIndex(
				Some(&self.projections),
				0,
				1,
			);
			enc.setBuffer_offset_atIndex(Some(&self.nodes), 0, 2);
			enc.setBuffer_offset_atIndex(
				Some(&self.transitions),
				0,
				3,
			);
		}
		enc.dispatchThreadgroups_threadsPerThreadgroup(
			MTLSize {
				width: num_pattern_blocks,
				height: leaves_end as usize,
				depth: 1,
			},
			MTLSize {
				width: block_size,
				height: 4,
				depth: 1,
			},
		);
		enc.endEncoding();
		Ok(())
	}

	/// Encode the main `propose` kernel onto `cmd`.
	fn encode_update_all(
		&self,
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
		leaves_end: u32,
		internals_start: u32,
	) -> Result<()> {
		let block_size: u32 = 16 * 4;
		let num_pattern_blocks =
			(self.num_patterns * 4).div_ceil(block_size);

		let enc = self.compute_encoder(cmd)?;

		enc.setComputePipelineState(&self.propose_pipeline);
		unsafe {
			enc.setBuffer_offset_atIndex(Some(&self.leaves), 0, 0);
			enc.setBuffer_offset_atIndex(
				Some(&self.projections),
				0,
				1,
			);
			enc.setBuffer_offset_atIndex(Some(&self.scales), 0, 2);
			enc.setBuffer_offset_atIndex(
				Some(&self.scale_sums),
				0,
				3,
			);
			self.set_u32(&enc, self.num_updated_nodes, 4);
			enc.setBuffer_offset_atIndex(Some(&self.nodes), 0, 5);
			enc.setBuffer_offset_atIndex(
				Some(&self.children),
				0,
				6,
			);
			enc.setBuffer_offset_atIndex(
				Some(&self.transitions),
				0,
				7,
			);
			self.set_u32(&enc, leaves_end, 8);
			self.set_u32(&enc, internals_start, 9);
		}
		enc.dispatchThreadgroups_threadsPerThreadgroup(
			MTLSize {
				width: num_pattern_blocks as usize,
				height: 1,
				depth: 1,
			},
			MTLSize {
				width: block_size as usize,
				height: 1,
				depth: 1,
			},
		);
		enc.endEncoding();
		Ok(())
	}

	/// Encode the root likelihood kernel onto `cmd`.
	fn encode_update_likelihoods(
		&self,
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
		root: u32,
		(left_child, right_child): (u32, u32),
		frequencies: [f32; 4],
	) -> Result<()> {
		let block_size: u32 = 32;
		let num_pattern_blocks = self.num_patterns.div_ceil(block_size);

		let enc = self.compute_encoder(cmd)?;

		enc.setComputePipelineState(&self.update_likelihoods_pipeline);
		unsafe {
			enc.setBuffer_offset_atIndex(
				Some(&self.projections),
				0,
				0,
			);
			enc.setBuffer_offset_atIndex(
				Some(&self.likelihoods),
				0,
				1,
			);
			enc.setBuffer_offset_atIndex(Some(&self.scales), 0, 2);
			enc.setBuffer_offset_atIndex(
				Some(&self.scale_sums),
				0,
				3,
			);
			self.set_u32(&enc, root, 4);
			self.set_u32(&enc, left_child, 5);
			self.set_u32(&enc, right_child, 6);
			enc.setBytes_length_atIndex(
				NonNull::new(
					frequencies.as_ptr() as *mut c_void
				)
				.unwrap(),
				size_of_val(&frequencies),
				7,
			);
		}
		enc.dispatchThreadgroups_threadsPerThreadgroup(
			MTLSize {
				width: num_pattern_blocks as usize,
				height: 1,
				depth: 1,
			},
			MTLSize {
				width: block_size as usize,
				height: 1,
				depth: 1,
			},
		);
		enc.endEncoding();
		Ok(())
	}

	/// Encode the copy_projections kernel (accept or reject) onto `cmd`.
	fn encode_copy_projections(
		&mut self,
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
		accept: bool,
	) -> Result<()> {
		let num_updated = self.num_updated_nodes + 1;
		let num_pattern = self.num_patterns.div_ceil(128);
		let grid_dim_y = num_updated.div_ceil(128);

		let enc = self.compute_encoder(cmd)?;

		enc.setComputePipelineState(&self.copy_projections_pipeline);
		unsafe {
			if accept {
				enc.setBuffer_offset_atIndex(
					Some(&self.projections),
					0,
					0,
				);
				enc.setBuffer_offset_atIndex(
					Some(&self.projections_backup),
					0,
					1,
				);
				enc.setBuffer_offset_atIndex(
					Some(&self.scales),
					0,
					2,
				);
				enc.setBuffer_offset_atIndex(
					Some(&self.scales_backup),
					0,
					3,
				);
			} else {
				enc.setBuffer_offset_atIndex(
					Some(&self.projections_backup),
					0,
					0,
				);
				enc.setBuffer_offset_atIndex(
					Some(&self.projections),
					0,
					1,
				);
				enc.setBuffer_offset_atIndex(
					Some(&self.scales_backup),
					0,
					2,
				);
				enc.setBuffer_offset_atIndex(
					Some(&self.scales),
					0,
					3,
				);
			}
			self.set_u32(&enc, num_updated, 4);
			enc.setBuffer_offset_atIndex(Some(&self.nodes), 0, 5);
		}
		// Mirror the CUDA 3-D grid: x = pattern blocks, y = node blocks, z = 128
		enc.dispatchThreadgroups_threadsPerThreadgroup(
			MTLSize {
				width: num_pattern as usize,
				height: grid_dim_y as usize,
				depth: 128,
			},
			MTLSize {
				width: 128,
				height: 1,
				depth: 1,
			},
		);
		enc.endEncoding();
		Ok(())
	}

	/// Encode a `scale_sums` ↔ `scale_sums_backup` blit copy onto `cmd`.
	fn encode_blit_scale_sums(
		&self,
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
		accept: bool,
	) -> Result<()> {
		let enc = cmd
			.blitCommandEncoder()
			.context("Metal: blitCommandEncoder")?;

		let bytes = self.num_patterns as usize * size_of::<u32>();
		let (src, dst) = if accept {
			(&self.scale_sums, &self.scale_sums_backup)
		} else {
			(&self.scale_sums_backup, &self.scale_sums)
		};
		// SAFETY: buffers are correctly sized and non-overlapping.
		unsafe {
			enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                src, 0, dst, 0, bytes,
            );
		}
		enc.endEncoding();
		Ok(())
	}

	/// Check that the last committed command buffer completed without error.
	fn check_completion(
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
	) -> Result<()> {
		if let Some(err) = cmd.error() {
			return Err(anyhow!(
				"Metal command buffer error: {err:?}"
			));
		}
		Ok(())
	}

	// ── Low-level helpers ──────────────────────────────────────────────────

	fn command_buffer(
		&self,
	) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
		self.queue.commandBuffer().context("Metal: commandBuffer")
	}

	fn compute_encoder(
		&self,
		cmd: &ProtocolObject<dyn MTLCommandBuffer>,
	) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
		cmd.computeCommandEncoder()
			.context("Metal: computeCommandEncoder")
	}

	/// Pass a `u32` scalar via `setBytes` (avoids allocating a small buffer).
	fn set_u32(
		&self,
		enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
		value: u32,
		index: usize,
	) {
		// SAFETY: `value` is on the stack for the duration of the call;
		// Metal copies the bytes before returning.
		unsafe {
			enc.setBytes_length_atIndex(
				NonNull::new(std::ptr::addr_of!(value)
					as *mut c_void)
				.unwrap(),
				size_of::<u32>(),
				index,
			);
		}
	}

	pub fn new(
		pattern_weights: Vec<u32>,
		leaves: Vec<u8>,
		scale_ln: u32,
		_device_index: usize,
	) -> Result<Self> {
		let device = MTLCreateSystemDefaultDevice()
			.context("Metal device not available")?;
		let queue = device
			.newCommandQueue()
			.context("Metal: newCommandQueue")?;

		// f32 scale constants (matching the f32 GPU buffers above).
		// Clamp to f32 representable range: scale_ln > 88 would overflow
		// f32::MAX ≈ e^88.7, so e^300 = inf.  Clamping scale_mult to f32::MAX
		// effectively disables rescaling for those very-large scale_ln values
		// (the threshold becomes 0.0f so the branch never fires).
		let scale_threshold = (-(scale_ln as f32)).exp();
		let scale_mult = (scale_ln as f32).exp().min(f32::MAX);

		let num_patterns = pattern_weights.len();
		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		// Bake constants into the shader source.
		// Use explicit decimal notation so Metal always parses them as floats.
		let full_src =
			format!("#define NUM_PATTERNS {num_patterns}u\n\
             #define NUM_LEAVES   {num_leaves}u\n\
             #define SCALE_LN     {scale_ln}u\n\
             #define SCALE_THRESHOLD {scale_threshold:e}f\n\
             #define SCALE_MULT    {scale_mult:e}f\n\
             {METAL_SRC}",);
		let src_ns = NSString::from_str(&full_src);
		let opts = MTLCompileOptions::new();
		// Metal 3.0+ enables double-precision arithmetic on Apple silicon
		// (GPU family apple7+) and recent AMD GPUs.
		opts.setLanguageVersion(MTLLanguageVersion::Version3_0);
		let library = device
			.newLibraryWithSource_options_error(
				&src_ns,
				Some(&opts),
			)
			.map_err(|e| {
				anyhow!(
					"Metal shader compilation failed: {e:?}"
				)
			})?;

		let propose_pipeline =
			compile_pipeline(&device, &library, "propose")?;
		let update_leaves_pipeline =
			compile_pipeline(&device, &library, "update_leaves")?;
		let update_likelihoods_pipeline = compile_pipeline(
			&device,
			&library,
			"update_likelihoods",
		)?;
		let copy_projections_pipeline = compile_pipeline(
			&device,
			&library,
			"copy_projections",
		)?;

		let leaves_buf = upload_buf::<u8>(&device, &leaves)?;
		// GPU buffers use f32: Apple GPU f64 throughput is ~1/32 of f32.
		let projections =
			alloc_buf::<GpuRow>(&device, num_nodes * num_patterns)?;
		let projections_backup =
			alloc_buf::<GpuRow>(&device, num_nodes * num_patterns)?;
		let likelihoods = alloc_buf::<f32>(&device, num_patterns)?;
		let children = alloc_buf::<u32>(&device, num_internals * 2)?;
		let transitions =
			alloc_buf::<GpuTransition>(&device, num_edges)?;
		let nodes = alloc_buf::<u32>(&device, num_nodes)?;
		let scales =
			alloc_buf::<u8>(&device, num_nodes * num_patterns)?;
		let scales_backup =
			alloc_buf::<u8>(&device, num_nodes * num_patterns)?;
		let scale_sums = alloc_buf::<u32>(&device, num_patterns)?;
		let scale_sums_backup =
			alloc_buf::<u32>(&device, num_patterns)?;

		Ok(Self {
			_device: device,
			queue,

			propose_pipeline,
			update_leaves_pipeline,
			update_likelihoods_pipeline,
			copy_projections_pipeline,

			leaves: leaves_buf,
			projections,
			projections_backup,
			likelihoods,
			children,
			transitions,
			nodes,

			scales,
			scales_backup,
			scale_sums,
			scale_sums_backup,

			pattern_weights,
			num_patterns: num_patterns as u32,
			num_updated_nodes: 0,
		})
	}
}

fn compile_pipeline(
	device: &ProtocolObject<dyn MTLDevice>,
	library: &ProtocolObject<dyn MTLLibrary>,
	name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
	let name_ns = NSString::from_str(name);
	let func = library
		.newFunctionWithName(&name_ns)
		.ok_or_else(|| anyhow!("Metal: function '{name}' not found"))?;
	let pipeline = device
		.newComputePipelineStateWithFunction_error(&func)
		.map_err(|e| {
			anyhow!("Metal: pipeline '{name}' failed: {e:?}")
		})?;
	Ok(pipeline)
}
