"""Stock scene energy metadata never changes native model data."""
import pathlib,sys,unittest,os
import numpy as np
import warp as wp
import newton
from gpu.dat_stock_scene import attach_stock_ipc,model_array_snapshot,rest_edge_coefficients


def model():
 b=newton.ModelBuilder(gravity=(0,0,0))
 b.add_particles([[0,0,0],[2.75,0,0],[0,1,0],[0,-1,0]],[[0,0,0]]*4,[.1]*4,flags=[0,1,1,1],radius=[0]*4)
 b.add_triangles([0,1],[1,0],[2,3],tri_ke=[1000]*2,tri_ka=[1000]*2,tri_kd=[.0002]*2)
 b.add_edges([2],[3],[0],[1],rest=[0],edge_ke=[.001],edge_kd=[.01])
 b.color();return b.finalize(device='cpu')


class BridgeTests(unittest.TestCase):
 def test_metadata_only_hard_flags_materials_rest_and_colors_unchanged(self):
  m=model();before=model_array_snapshot(m);receipt=attach_stock_ipc(m,gs_predictor='none');self.assertEqual(before,model_array_snapshot(m));self.assertTrue(receipt['existing_arrays_unchanged'])
  self.assertEqual(m.particle_flags.numpy()[0]&1,0);self.assertFalse(np.any(m.ipc.pinned.numpy()));self.assertEqual(m.ipc.params['kpin'],0)
  self.assertIs(m.ipc.tri_inverse,m.tri_poses);self.assertIs(m.ipc.tri_area,m.tri_areas);self.assertIs(m.ipc.original_mass,m.particle_mass)
  self.assertTrue(np.all(m.tri_materials.numpy()[:,2]>0));self.assertGreater(m.edge_bending_properties.numpy()[0,1],0)
 def test_coefficient_once_and_effective_bending_matched_in_separate_array(self):
  m=model();raw=m.edge_bending_properties.numpy().copy();receipt=attach_stock_ipc(m)
  self.assertEqual(float(m.ipc.edge_coefficient.numpy()[0]),2.75)
  expected=float(raw[0,0])*float(m.edge_rest_length.numpy()[0])/(2*2.75)
  self.assertEqual(m.ipc.dat_bending_stiffness.numpy()[0],np.float32(expected))
  np.testing.assert_array_equal(raw,m.edge_bending_properties.numpy())
  own=np.float32(np.float32(2)*m.ipc.dat_bending_stiffness.numpy()[0]*m.ipc.edge_coefficient.numpy()[0])
  native=np.float32(raw[0,0]*m.edge_rest_length.numpy()[0]);self.assertEqual(own.tobytes(),native.tobytes())
  self.assertEqual(receipt['bending_mapping'],'matched-effective')
 def test_direct_lame_triangle_mapping_has_no_subtraction_or_thickness_scaling(self):
  m=model();original=m.tri_materials.numpy().copy();receipt=attach_stock_ipc(m)
  np.testing.assert_array_equal(m.ipc.dat_triangle_materials.numpy(),original[:,:2]);np.testing.assert_array_equal(m.tri_materials.numpy(),original)
  self.assertEqual(m.ipc.params['lambda'],1000);self.assertTrue(all(x==3000 for x in receipt['small_strain_uniaxial_modulus_stock_and_own']))
 def test_native_order_and_invalid_rest_records(self):
  c,order=rest_edge_coefficients([[0,1,2],[1,0,3]],[1.375,1.375],[[2,3,0,1]],[2.75]);self.assertEqual(c[0],2.75);np.testing.assert_array_equal(order,[[0,1,2,3]])
  with self.assertRaisesRegex(ValueError,'rest measure'):rest_edge_coefficients([[0,1,2]],[0],[[-1,2,0,1]],[1])
 def test_explicit_same_numeric_diagnostic_and_duplicate_attachment_rejection(self):
  m=model();attach_stock_ipc(m,bending_mapping='same-numeric');self.assertEqual(m.ipc.dat_bending_stiffness.numpy()[0],m.edge_bending_properties.numpy()[0,0])
  with self.assertRaisesRegex(ValueError,'unmodified'):attach_stock_ipc(m)

if __name__=='__main__':unittest.main()
