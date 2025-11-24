/*!
**************************************************************************
* DT-MIL
* Copyright (c) 2021 Tencent. All Rights Reserved.
**************************************************************************
* Modified from Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
**************************************************************************
*/

#include "ms_deform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
