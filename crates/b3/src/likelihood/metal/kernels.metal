#include <metal_stdlib>
using namespace metal;

typedef unsigned char u8;
typedef unsigned int u32;
typedef float f32;
typedef float4 f32x4;

// NUM_PATTERNS, NUM_LEAVES, SCALE_LN, SCALE_THRESHOLD, SCALE_MULT
// are known at compile time.

#define BLOCK_SIZE (16 * 4)

#define idx(edge) ((edge) * NUM_PATTERNS + pattern)
#define sidx(edge) (((edge) * NUM_PATTERNS + pattern) * 4 + sub)

#define CALCULATE_LEAF_PROJECTION \
    u8 leaf = leaves[idx(nodes[i])]; \
    f32 projection = 0.0f; \
    f32x4 tv = transitions[i * 4 + sub]; \
    if (leaf == 0x01)      projection = tv.x; \
    else if (leaf == 0x02) projection = tv.y; \
    else if (leaf == 0x04) projection = tv.z; \
    else if (leaf == 0x08) projection = tv.w; \
    else                   projection = 1.0f; \
    projections[sidx(nodes[i])] = projection;

// Grid:  (ceil(num_patterns/16), leaves_end, 1)
// Block: (16, 4, 1)
kernel void update_leaves(
    device const u8*    leaves      [[buffer(0)]],
    device f32*         projections [[buffer(1)]],
    device const u32*   nodes       [[buffer(2)]],
    device const f32x4* transitions [[buffer(3)]],

    uint3 tpg  [[thread_position_in_threadgroup]],
    uint3 tgpg [[threadgroup_position_in_grid]]
) {
    u32 pattern = tgpg.x * 16 + tpg.x;
    if (pattern >= NUM_PATTERNS) return;
    u32 sub = tpg.y;
    u32 i   = tgpg.y;

    CALCULATE_LEAF_PROJECTION
}

// Grid:  (ceil(num_patterns*4 / BLOCK_SIZE), 1, 1)
// Block: (BLOCK_SIZE, 1, 2)
kernel void propose(
    device const u8*    leaves              [[buffer(0)]],
    device f32*         projections         [[buffer(1)]],
    device u8*          scales              [[buffer(2)]],
    device u32*         scale_sums          [[buffer(3)]],
    device const u32*   num_updated_buf     [[buffer(4)]],
    device const u32*   nodes               [[buffer(5)]],
    device const u32*   children            [[buffer(6)]],
    device const f32x4* transitions         [[buffer(7)]],
    device const u32*   leaves_end_buf      [[buffer(8)]],
    device const u32*   internals_start_buf [[buffer(9)]],

    uint tpg_x  [[thread_position_in_threadgroup]],
    uint tgpg_x [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    u32 pattern = (tgpg_x * BLOCK_SIZE + tpg_x) / 4;
    if (pattern >= NUM_PATTERNS) return;

    u32 sub = tpg_x % 4;

    u32 num_updated = *num_updated_buf;
    u32 leaves_end = *leaves_end_buf;
    u32 internals_start = *internals_start_buf;

    ushort base = (simd_lane / 4) * 4;

    for (u32 i = 0; i < leaves_end; i++) {
        CALCULATE_LEAF_PROJECTION
    }

    u32 scale_sum = scale_sums[pattern];

    for (u32 i = internals_start; i < num_updated; i++) {
        u32 left_edge  = children[(i - internals_start) * 2];
        u32 right_edge = children[(i - internals_start) * 2 + 1];
        u32 this_edge  = nodes[i];
        u32 scale_idx  = idx(this_edge);
        u32 old_scale  = scales[scale_idx];

        f32 l_likelihood = projections[sidx(left_edge)] *
                           projections[sidx(right_edge)];

        f32 v0 = simd_shuffle(l_likelihood, base + 0);
        f32 v1 = simd_shuffle(l_likelihood, base + 1);
        f32 v2 = simd_shuffle(l_likelihood, base + 2);
        f32 v3 = simd_shuffle(l_likelihood, base + 3);

        u32 should_scale =
            (v0 < SCALE_THRESHOLD &&
             v1 < SCALE_THRESHOLD &&
             v2 < SCALE_THRESHOLD &&
             v3 < SCALE_THRESHOLD) ? 1u : 0u;

        if (should_scale) {
            v0 *= SCALE_MULT;
            v1 *= SCALE_MULT;
            v2 *= SCALE_MULT;
            v3 *= SCALE_MULT;
        }

        if (sub == 0 && should_scale != old_scale) {
            scales[scale_idx] = (u8)should_scale;
            if (old_scale == 0) {
                scale_sum += SCALE_LN;
            } else {
                scale_sum -= SCALE_LN;
            }
        }

        f32x4 likelihood = f32x4(v0, v1, v2, v3);

        projections[sidx(this_edge)] = dot(transitions[i * 4 + sub], likelihood);
    }

    if (sub == 0) {
        scale_sums[pattern] = scale_sum;
    }
}

// Grid:  (ceil(num_patterns / 32), 1, 1)
// Block: (32, 1, 1)
kernel void update_likelihoods(
    device const f32x4* projections     [[buffer(0)]],
    device f32*         likelihoods     [[buffer(1)]],
    device u8*          scales          [[buffer(2)]],
    device u32*         scale_sums      [[buffer(3)]],
    device const u32*   root_buf        [[buffer(4)]],
    device const u32*   left_buf        [[buffer(5)]],
    device const u32*   right_buf       [[buffer(6)]],
    device const f32x4* frequencies_buf [[buffer(7)]],

    uint gid [[thread_position_in_grid]]
) {
    u32 pattern = gid;
    if (pattern >= NUM_PATTERNS) return;

    u32 root        = *root_buf;
    u32 left_child  = *left_buf;
    u32 right_child = *right_buf;
    f32x4 freq      = *frequencies_buf;

    f32x4 left  = projections[idx(left_child)];
    f32x4 right = projections[idx(right_child)];

    f32x4 lk = left * right * freq;
    likelihoods[pattern] = log(lk.x + lk.y + lk.z + lk.w);

    u32 root_scale = idx(root);
    if (scales[root_scale]) {
        scales[root_scale] = 0;
        scale_sums[pattern] -= SCALE_LN;
    }
}

// Grid:  (ceil(num_patterns / 128), ceil(num_updated / 128), 128)
// Block: (128, 1, 1)
kernel void copy_projections(
    device const f32x4* p_src           [[buffer(0)]],
    device f32x4*       p_dst           [[buffer(1)]],
    device const u8*    s_src           [[buffer(2)]],
    device u8*          s_dst           [[buffer(3)]],
    device const u32*   num_updated_buf [[buffer(4)]],
    device const u32*   nodes           [[buffer(5)]],

    uint3 gid [[thread_position_in_grid]]
) {
    u32 pattern = gid.x;
    if (pattern >= NUM_PATTERNS) return;

    u32 i = gid.y * 128 + gid.z;
    if (i >= *num_updated_buf) return;

    u32 proj_idx = idx(nodes[i]);
    p_dst[proj_idx] = p_src[proj_idx];
    s_dst[proj_idx] = s_src[proj_idx];
}
