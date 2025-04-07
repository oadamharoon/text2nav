def setup_scene(Self):
  self.simulation_context = SimulationContext.instance()
  world = self.get_world()
  world.scene.add_default_ground_plane()
  stage = omni.usd.get_context().get_stage()
  _setup_env(stage)
  
def _setup_env(self, stage):
  # Create a parent Xform for the scan object
  parent_path = "/World"
  parent_prim = stage.DefinePrim(parent_path, "Xform")
        
  # Load the YCB pitcher as a reference under the parent
  object_path = f"{parent_path}/Hospital"
  asset_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Environments/Hospital/hospital.usd"
  object_prim = stage.DefinePrim(object_path, "Xform")
  object_prim.GetReferences().AddReference(asset_path)

