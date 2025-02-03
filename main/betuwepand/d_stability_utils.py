import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from scipy.interpolate import interp1d

from geolib.models.dstability import DStabilityModel
from geolib.models.dstability.internal import PersistablePoint


def adapt_stability_file(stabilitytemplate=r"Cross-Section 49",
                         stabilityfilename=r"Evidence Cross-Section 49",
                         directory=Path(r'.\calculations'),
                         waterlevel=4):
    
    dm = DStabilityModel()
    
    dm.parse(Path.joinpath(directory,stabilitytemplate+'.stix'))
    
    adapt_waternet(dm,stabilitytemplate,directory,waterlevel)
    
    dm.serialize(Path.joinpath(directory,stabilityfilename+'.stix'))    
    
    return


def adapt_waternet(dm,
                   stabilitytemplate,
                   directory,
                   waterlevel):
    r"""
    """
    
    waterpressures = Path.joinpath(directory,r'Water Pressures '+stabilitytemplate+'.xlsx')
    
    for i,name in zip(range(3),['PL','HL1','HL2']):
        
        temp = []
        
        temp_ = pd.read_excel(waterpressures, sheet_name = name )
        
        for j in range(len(temp_)):
            
            f = interp1d([temp_.iloc[0].Z1,temp_.iloc[0].Z2],[temp_.iloc[j].Z1,temp_.iloc[j].Z2],fill_value="extrapolate")
            
            temp.append(PersistablePoint(X=temp_.iloc[j].X, Z=f(waterlevel)))
        
        dm.datastructure.waternets[1].HeadLines[i].Points = temp
    
    return dm


def get_surface_line(dm):
    
    polygons = []
    
    for i in dm.datastructure.geometries[0].Layers:
    
        x = np.zeros(len(i.Points)+1)
        z = np.zeros(len(i.Points)+1)
    
        for j, n in zip(i.Points, range(len(i.Points))):
            
            x[n] = j.X
            z[n] = j.Z
        
        x[-1] = i.Points[0].X    
        z[-1] = i.Points[0].Z
        
        polygons.append(Polygon(list(zip(x,z))))
    
    boundary = np.asarray(cascaded_union(polygons).boundary.xy).T
    
    i = np.where(boundary[:,0]==np.min(boundary[:,0]))
    i = np.delete(i,np.where(boundary[i,1]==np.max(boundary[i,1]))[1][0])
    
    i2 = np.where(boundary[:,0]==np.max(boundary[:,0]))
    i2 = np.delete(i2,np.where(boundary[i2,1]==np.max(boundary[i2,1]))[1][0])
    
    i = np.concatenate((i,i2),axis=0)
    
    surface_line = np.delete(boundary,i,axis=0)   
    
    return surface_line


def design_big_bags(realizations,
                    dm,
                    number_of_big_bags):
    
    surface_line = get_surface_line(dm)
    
    uittredepunt = []

    for i in range(len(realizations.Realization.unique())):
        
        sel = realizations[realizations.Realization == i]
        
        if (float(sel[sel.Variable=='SF'].PhysicalValue) < 1) & (float(sel[sel.Variable=='EV_SF'].PhysicalValue) > 1):
            
            temp = PlotSlipPlane(xcl = float(sel[sel.Variable=='LeftCenter.left_X'].PhysicalValue),
                                 zcl = float(sel[sel.Variable=='LeftCenter.left_Z'].PhysicalValue),
                                 xcr = float(sel[sel.Variable=='RightCenter.right_X'].PhysicalValue),
                                 zcr = float(sel[sel.Variable=='RightCenter.right_Z'].PhysicalValue),
                                 t   = float(sel[sel.Variable=='TL'].PhysicalValue),
                                 cs  = surface_line,
                                 color = 'r')
            
            uittredepunt.append(temp[0][-1])
        
    f=interp1d(surface_line[:,0],surface_line[:,1])
    
    big_bag_locations = []    
    
    for i in range(number_of_big_bags):

        x = np.array([np.min(uittredepunt)-(i+1),
                      np.min(uittredepunt)-i,
                      np.min(uittredepunt)-i,
                      np.min(uittredepunt)-(i+1)])
    
        z = f(x)
        z[-2:] = z[-2:]+1
        
        big_bag_locations.append(Polygon(list(zip(x,z))))
    
    return big_bag_locations


def get_soilcolor(dm, layer_id):
    color = None
    name = None
    for soillayer in dm.input.soillayers[0].SoilLayers:
        if soillayer.LayerId == layer_id:
            soil_id = soillayer.SoilId
    for soil in dm.input.soilvisualizations.SoilVisualizations:
        if soil.SoilId == soil_id:
            color = soil.Color.replace("#80", "#")
    for soil in dm.input.soils.Soils:
        if soil.Id == soil_id:
            name = soil.Code
    return color,name


def get_phreatic_line(dm):
    
    PL = []
    
    for headline in dm.input.waternets[-1].HeadLines:
        if headline.Id == dm.input.waternets[-1].PhreaticLineId:
            for points in headline.Points:
                PL.append([points.X,points.Z])
                
    return np.asarray(PL)


def get_cross_section(stability_model:str, water_levels: list, location:list) -> dict:
    r"""

    Function that gets cross-section from the D-Stability model and returns a dictionary with the cross-section graph.

    Parameters
    ----------
    water_levels : list
        List containing dictionaries with coordinates of water levels ('s', 'z', 'max_wl', 'name').
    stability_model : str
        Path to the stability model.

    Returns
    -------
    dict
        Dictionary containing the cross-section graph. With the following
        keys: 'Soils', 'SurfaceLine', 'WaterLevels'.

    """


    cross_section = {}
    dm = DStabilityModel()
    dm.parse(Path(stability_model))
    cross_section['Coordinates'] = location

    cross_section['Soils'] = []

    for soil_layer in dm.datastructure.geometries[0].Layers:

        x = np.zeros(len(soil_layer.Points) + 1)
        z = np.zeros(len(soil_layer.Points) + 1)

        for j, n in zip(soil_layer.Points, range(len(soil_layer.Points))):
            x[n] = j.X
            z[n] = j.Z

        x[-1] = soil_layer.Points[0].X
        z[-1] = soil_layer.Points[0].Z

        color, name = get_soilcolor(dm, soil_layer.Id)

        if name == 'C':
            color = 'green'
        elif name == 'P':
            color = 'brown'
        elif name == 'S':
            color = 'yellow'

        cross_section['Soils'].append({'x': x.tolist(), 'z': z.tolist(), 'color': color})

    # get the surface line
    cross_section['SurfaceLine'] = get_surface_line(dm).tolist()
    cross_section['WaterLevels'] = ''
    return cross_section

