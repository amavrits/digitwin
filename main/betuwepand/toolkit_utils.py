import os
import sys
from pathlib import Path

sys.path.append(r"C:\Program Files (x86)\Deltares\Probabilistic Toolkit\Python")
# from toolkit_model import ToolKit


def run_ptk_calculation(directory, filename, ptk=None, proj=None):

    if not ptk and not proj:

        ptk = ToolKit()

        proj = ptk.load(Path.joinpath(directory, filename + '.tkx'))

    elif not ptk or not proj:

        print('error')

    proj.run()

    pf = proj.design_point.probability_failure
    conv = proj.design_point.convergence

    ptk.save(Path.joinpath(directory, filename + '_temp.tkx'))

    os.system("taskkill /f /im  Deltares.Probabilistic.Server.exe")

    return pf, conv

# pf = run_ptk_calculation(Path(r'C:\Users\meer_an\repositories\purmerfloodingprediction\scenarios\purmer\work_folder'), 'prob')
