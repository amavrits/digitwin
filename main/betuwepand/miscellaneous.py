from shapely.geometry import LineString, Point
import math
from pathlib import Path
from geolib.models.dstability import DStabilityModel
import subprocess
from shapely.geometry import Point
from geolib.models.dstability.internal import PersistablePoint
from draw_slip_plane import draw_lift_van_slip_plane
from d_stability_utils import get_surface_line
from toolkit_utils import run_ptk_calculation

STABILITY_CONSOLE = r"C:\Program Files (x86)\Deltares\D-GEO Suite\D-Stability 2024.01\bin\D-Stability Console.exe"


def cs2linestring(df):

    cs = []

    for i in range(len(df)):

        if isinstance(df.iloc[i].RDX_start, float) and (math.isnan(df.iloc[i].RDX_start)==False):

            cs.append(LineString(
                [Point(df.iloc[i].RDX_start, df.iloc[i].RDY_start),
                 Point(df.iloc[i].RDX_end,	df.iloc[i].RDY_end)]))

    return cs


def adapt_waternet(dm,PL,HL):
    r"""
    """

    temp = []

    for j in range(len(PL)):

        temp.append(PersistablePoint(X=PL[j, 0], Z=PL[j, 1]))

    dm.datastructure.waternets[0].HeadLines[0].Points = temp

    dm.datastructure.waternets[0].HeadLines[1].Points = [PersistablePoint(X=HL[0, 0], Z=HL[0, 1])]

    return dm

def change_waterline(model, name , points):
    r"""
    """
    for i in range(len(model.datastructure.waternets)):
        headlines = model.datastructure.waternets[i].HeadLines
        for j in range(len(headlines)):
            if headlines[j].Label == name:
                # points to geolib points
                model.datastructure.waternets[i].HeadLines[j].Points = [PersistablePoint(X=point[0], Z=point[1]) for point in list(points)]
    return model


def run_scenario(parsed_file: str, scenario: dict, probabilistic: bool = False):
    r"""
    Run a scenario in the D-Stability model and return the results.

    Parameters
    ----------
    parsed_file : str
        Path to the parsed file.
    scenario : dict
        Dictionary containing the scenario details.
    probabilistic : bool
        If True, run the probabilistic calculation.

    Returns
    -------
    dict
        Dictionary containing the scenario details and the results.

    """
    dm = DStabilityModel()
    dm.parse(Path(parsed_file))
    surface_line = get_surface_line(dm)
    # Read PL and HL per scenario and pass to D-Stability model
    dm = change_waterline(dm, 'HL 1', scenario['HL'])
    dm = change_waterline(dm, 'HL 2', scenario['PL'])

    # Calculate FoS and collect slip plane details.
    StabilityConsole = STABILITY_CONSOLE
    dm.serialize(Path(r'.\work_folder\test_Case.stix'))
    FileName = r".\work_folder\test_Case.stix"
    cmd = ('"' + StabilityConsole + '" "' + FileName + '"')
    subprocess.call(cmd, shell=True)
    dm2 = DStabilityModel()
    dm2.parse(Path('./work_folder/test_Case.stix'))
    # FoS = dm2.get_result().FactorOfSafety
    FoS = [output.FactorOfSafety for output in dm2.output]
    slip_plane = draw_lift_van_slip_plane(x_center_left=dm2.get_result().get_slipcircle_output().x_left,
                                          z_center_left=dm2.get_result().get_slipcircle_output().z_left,
                                          x_center_right=dm2.get_result().get_slipcircle_output().x_right,
                                          z_center_right=dm2.get_result().get_slipcircle_output().z_right,
                                          tangent_line=dm2.get_result().get_slipcircle_output().z_tangent,
                                          surface_line={'s': surface_line[:, 0], 'z': surface_line[:, 1]})
    scenario['FoS'] = FoS
    scenario['Slip plane'] = [list(slip_plane[0]), list(slip_plane[1])]
    # date to string
    scenario['date'] = str(scenario['date'])

    # Calculate P(F) using the PTK. If the calculation in not performe, pass P(F) = nan
    if probabilistic:
        pf, conv = run_ptk_calculation(Path('./work_folder'), 'prob')
        scenario['Pf'] = pf
        scenario['Convergence'] = conv
    else:
        scenario['Pf'] = 'nan'
        scenario['Convergence'] = 'nan'

    return scenario
