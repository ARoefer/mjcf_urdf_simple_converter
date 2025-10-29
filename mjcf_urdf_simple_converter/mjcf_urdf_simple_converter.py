import mujoco
import mujoco._structs
import numpy
from xml.etree import ElementTree as ET
from xml.dom import minidom
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
from stl import mesh
from typing import Union
import os

def array2str(arr):
    return " ".join([str(x) for x in arr])

def create_body(xml_root, name, inertial_pos, inertial_rpy, mass, ixx, iyy, izz):
    """
    create a body with given mass and inertia
    """
    # create XML element for this body
    body = ET.SubElement(xml_root, 'link', {'name': name})

    # add inertial element
    inertial = ET.SubElement(body, 'inertial')
    ET.SubElement(inertial, 'origin', {'xyz': array2str(inertial_pos), 'rpy': array2str(inertial_rpy)})
    ET.SubElement(inertial, 'mass', {'value': str(mass)})
    ET.SubElement(inertial, 'inertia', {'ixx': str(ixx), 'iyy': str(iyy), 'izz': str(izz),
                                        'ixy': "0", 'ixz': "0", 'iyz': "0"})
    return body

def create_dummy_body(xml_root, name):
    """
    create a dummy body with negligible mass and inertia
    """
    mass = 0.001
    mass_moi = mass * (0.001 ** 2)  # mass moment of inertia
    return create_body(xml_root, name, np.zeros(3), np.zeros(3), mass, mass_moi, mass_moi, mass_moi)
    

def create_joint(xml_root, name, parent, child, pos, rpy, joint_type : str, axis=None, jnt_range=None):
    """
    if axis and jnt_range is None, create a fixed joint. otherwise, create a revolute joint
    """
    # create joint element connecting this to parent
    jnt_element = ET.SubElement(xml_root, 'joint', {'type': joint_type, 'name': name})
    ET.SubElement(jnt_element, 'parent', {'link': parent})
    ET.SubElement(jnt_element, 'child', {'link': child})
    ET.SubElement(jnt_element, 'origin', {'xyz': array2str(pos), 'rpy': array2str(rpy)})
    if joint_type != 'fixed':
        if axis is not None:
            ET.SubElement(jnt_element, 'axis', {'xyz': array2str(axis)})
        if jnt_range is not None:
            ET.SubElement(jnt_element, 'limit', {'lower': str(jnt_range[0]), 'upper': str(jnt_range[1]), 'effort': "100", 'velocity': "100"})
    return jnt_element


def convert_subtree(root : ET.Element, model : mujoco.MjModel, id_or_name : Union[str, int],
                    output_dir : Path=None, asset_file_prefix : str="") -> ET.Element:
    # Convenience: Identify a root body either by its name or by the id
    if isinstance(id_or_name, int):
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)
        id = id_or_name
    else:
        if isinstance(model, mujoco._structs.MjModel):
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(id_or_name))
        else:
            id = model.body_name2id(id_or_name)
        child_name = id_or_name

    parent_id = model.body_parentid[id]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if isinstance(model, mujoco._structs.MjModel) else model.body_id2name(parent_id)

    # URDFs assume that the link origin is at the joint position, while in MJCF they can have user-defined values
    # this requires some conversion for the visual, inertial, and joint elements...
    # this is done by creating a dummy body with negligible mass and inertia at the joint position.
    parent_T_child = np.eye(4)
    parent_T_child[:3, :3] = Rotation.from_quat(model.body_quat[id], scalar_first=True).as_matrix() # [w, x, y, z]
    parent_T_child[:3,  3] = model.body_pos[id]

    # read inertial info
    mass = model.body_mass[id]
    inertia = model.body_inertia[id]
    child_P_inertia = model.body_ipos[id]
    child_Q_inertia = model.body_iquat[id]  # [w, x, y, z]
    child_RPY_inertia = Rotation.from_quat(child_Q_inertia, scalar_first=True).as_euler('xyz')
    # change to [x, y, z, w]

    # create child body
    body_element = create_body(root, child_name, child_P_inertia, child_RPY_inertia, mass, inertia[0], inertia[1], inertia[2])

    # read geom info and add it child body
    geomnum = model.body_geomnum[id]
    for geomnum_i in range(geomnum):
        geomid = model.body_geomadr[id] + geomnum_i
        if model.geom_type[geomid] != mujoco.mjtGeom.mjGEOM_MESH:
            # only support mesh geoms
            continue
        geom_dataid = model.geom_dataid[geomid]  # id of geom's mesh
        geom_pos = model.geom_pos[geomid]
        geom_quat = model.geom_quat[geomid]  # [w, x, y, z]
        # change to [x, y, z, w]
        geom_rpy = Rotation.from_quat(geom_quat, scalar_first=True).as_euler('xyz')
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, geom_dataid) if isinstance(model, mujoco._structs.MjModel) else model.geom_id2name(geom_dataid)

        # create visual element within body element
        if output_dir is not None:
            visual_element = ET.SubElement(body_element, 'visual', {'name': mesh_name})
            origin_element = ET.SubElement(visual_element, 'origin', {'xyz': array2str(geom_pos), 'rpy': array2str(geom_rpy)})
            geometry_element = ET.SubElement(visual_element, 'geometry')
            mesh_element = ET.SubElement(geometry_element, 'mesh', {'filename': f"{asset_file_prefix}converted_{mesh_name}.stl"})
            material_element = ET.SubElement(visual_element, 'material', {'name': 'white'})

            # create STL
            # the meshes in the MjModel seem to be different (have different pose) from the original STLs
            # so rather than using the original STLs, write them out from the MjModel
            # https://stackoverflow.com/questions/60066405/create-a-stl-file-from-a-collection-of-points
            vertadr = model.mesh_vertadr[geom_dataid]  # first vertex address
            vertnum = model.mesh_vertnum[geom_dataid]
            vert = model.mesh_vert[vertadr:vertadr+vertnum]
            normal = model.mesh_normal[vertadr:vertadr+vertnum]
            faceadr = model.mesh_faceadr[geom_dataid]  # first face address
            facenum = model.mesh_facenum[geom_dataid]
            face = model.mesh_face[faceadr:faceadr+facenum]
            data = np.zeros(facenum, dtype=mesh.Mesh.dtype)
            for i in range(facenum):
                data['vectors'][i] = vert[face[i]]
            m = mesh.Mesh(data, remove_empty_areas=False)
            mesh_save_path = output_dir / f"converted_{mesh_name}.stl"
            m.save(str(mesh_save_path))

    jntnum = model.body_jntnum[id]

    if child_name == "world":
        # there is no joint connecting the world to anything, since it is the root
        assert parent_name == "world"
        assert jntnum == 0
        return  # skip adding joint element or parent body

    if jntnum == 0:
        # We are not including the world frame in the URDF, so we can also not connect to it.
        if parent_name == 'world':
            return
        # No joints, create a fixed joint directly to parent
        create_joint(root,
                     f"{parent_name}2{child_name}_fixed",
                     parent_name, 
                     child_name,
                     parent_T_child[:3, 3],
                     Rotation.from_matrix(parent_T_child[:3, :3]).as_euler('xyz'), 'fixed')
    else:
        # For bodies with joints, create a chain of dummy bodies for each joint
        current_parent = parent_name

        parent_T_last = np.eye(4)

        # Process all joints for this body
        for j in range(jntnum):
            jntid = model.body_jntadr[id] + j
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jntid) if isinstance(model, mujoco._structs.MjModel) else model.joint_id2name(jntid)
            if jnt_name is None:
                # Generate a random name for the joint
                jnt_name = f"joint_{jntid}"
                print(f"WARNING: joint name for {jntid} is None (could happen for ball joints with >1DoF), using automatically generated name {jnt_name}")
            jnt_body_name = f"{jnt_name}_jointbody"
            
            # Joints are defined relative to the child body
            child_T_joint = np.eye(4)
            child_T_joint[:3, 3] = model.jnt_pos[jntid]
            parent_T_joint = parent_T_child @ child_T_joint

            # For URDF, we need to transform them into a chain
            last_T_joint  = np.linalg.inv(parent_T_last) @ parent_T_joint
            parent_T_last = parent_T_joint

            # Create dummy body for this joint
            create_dummy_body(root, jnt_body_name)
            
            is_limited = model.jnt_limited[jntid]
            if model.jnt_type[jntid] in {mujoco.mjtJoint.mjJNT_HINGE, 
                                         mujoco.mjtJoint.mjJNT_SLIDE}:
                jnt_range = model.jnt_range[jntid] if is_limited else [-10000, 10000]  # [min, max]
                child_V_axis = model.jnt_axis[jntid]  # [x, y, z]
                # Axes in URDF are in the joint's frame
                joint_V_axis = child_T_joint[:3, :3].T @ child_V_axis
                
                # Connect current parent to this joint body
                create_joint(root, jnt_name, current_parent, jnt_body_name, 
                            last_T_joint[:3, 3],
                            Rotation.from_matrix(last_T_joint[:3, :3]).as_euler('xyz'),
                            'revolute' if model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_HINGE else 'prismatic',
                            joint_V_axis,
                            jnt_range)
            else:
                # Handle other joint types (as fixed joints for now)
                print(f"doesn't support joint type {model.jnt_type[jntid]} from {parent_name} to {child_name}, treating as fixed joint...")
                
                create_joint(root, jnt_name,
                             current_parent,
                             jnt_body_name,
                             last_T_joint[:3, 3],
                             Rotation.from_matrix(last_T_joint[:3, :3]).as_euler('xyz'))
            
            current_parent = jnt_body_name
        
        # Connect last dummy body to child body with fixed joint
        # "bring back" the body coordinates to the child body frame
        child_T_last = np.linalg.inv(parent_T_last) @ parent_T_child
        create_joint(root,
                     f"{jnt_name}_offset",
                     current_parent,
                     child_name,
                     child_T_last[:3, 3],
                     Rotation.from_matrix(child_T_last[:3, :3]).as_euler('xyz'),
                     'fixed')


def object_to_urdf(model, object_name, robot_name=None, output_dir : Path=None, asset_file_prefix="") -> str:
    root = ET.Element('robot', {'name': object_name if robot_name is None else robot_name})

    root_id  = model.body_name2id(object_name)
    bodies   = {root_id}
    body_ids = np.arange(len(model.body_names))
    while True:
        connected = set(body_ids[np.isin(model.body_rootid, list(bodies))])
        if len(connected - bodies) == 0: # Did not find new connected bodies
            break
        bodies |= connected

    for on in np.asarray(model.body_names)[list(bodies)]:
        convert_subtree(root, model, on, output_dir, asset_file_prefix)
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")


def convert(mjcf_file, urdf_file, asset_file_prefix=""):
    """
    load MJCF file, parse it in mujoco and save it as URDF
    replicate just the kinematic structure, ignore most dynamics, actuators, etc.
    only works with mesh geoms
    https://mujoco.readthedocs.io/en/stable/APIreference.html#mjmodel
    http://wiki.ros.org/urdf/XML
    
    :param mjcf_file: path to existing MJCF file which will be loaded
    :param urdf_file: path to URDF file which will be saved
    :param asset_file_prefix: prefix to add to the stl file names (e.g. package://my_package/meshes/)
    """
    assert mjcf_file.endswith(".xml"), f"{mjcf_file=} should end with .xml"
    assert urdf_file.endswith(".urdf"), f"{urdf_file=} should end with .urdf"
    output_dir = os.path.dirname(urdf_file)
    assert os.path.exists(output_dir), f"{output_dir=} does not exist, please create it first"
    model = mujoco.MjModel.from_xml_path(mjcf_file)
    root = ET.Element('robot', {'name': "converted_robot"})
    root.append(ET.Comment('generated with mjcf_urdf_simple_converter (https://github.com/Yasu31/mjcf_urdf_simple_converter)'))

    for id in range(model.nbody):
        convert_subtree(root, model, id, output_dir, asset_file_prefix)
    
    # define white material
    material_element = ET.SubElement(root, 'material', {'name': 'white'})
    color_element = ET.SubElement(material_element, 'color', {'rgba': '1 1 1 1'})

    # write to file with pretty printing
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(urdf_file, "w") as f:
        f.write(xmlstr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mjcf_file', type=str)
    parser.add_argument('urdf_file', type=str)
    args = parser.parse_args()
    convert(args.mjcf_file, args.urdf_file)