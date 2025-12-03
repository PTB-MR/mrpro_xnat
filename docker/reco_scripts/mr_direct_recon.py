import os
import sys
from pathlib import Path
import ismrmrd
import torch
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import KData, CsmData, IData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd, KTrajectoryCartesian
from mrpro.operators import DictionaryMatchOp
from mrpro.operators.models import MonoExponentialDecay


def mr_direct_recon(fpath_input: Path | str, fpath_output: Path | str) -> int:
    """Direct (non-iterative) MR reconstruction using MRpro.

    Parameters
    ----------
        fpath_input
            Folder with raw data files
        fpath_output
            Output folder where reconstructed dicom images will be saved

    Returns
    ----------
        0 if everything went fine, 1 otherwise
    """
    fpath_input = fpath_input if isinstance(fpath_input, Path) else Path(fpath_input)
    fpath_output = (
        fpath_output if isinstance(fpath_output, Path) else Path(fpath_output)
    )

    print(f"Reading from {fpath_input}, writing into {fpath_output}")
    assert os.access(fpath_input, os.R_OK), (
        f"You don't have read permission in {fpath_input}"
    )
    assert os.access(fpath_output, os.W_OK), (
        f"You don't have write permission in {fpath_output}"
    )

    list_rawdata = sorted(fpath_input.glob("*.mrd"))
    if len(list_rawdata) == 0:
        print(f"No raw data found in {fpath_input}")
        return 1
    elif len(list_rawdata) > 1:
        print(
            f"Multiple raw data files found in {fpath_input}, only the first one will be processed: {list_rawdata[0]}"
        )

    # Very basic check if trajectory information is available in the ISMRMRD file
    with ismrmrd.File(list_rawdata[0], "r") as file:
        dataset = file[list(file.keys())[-1]]
        acquisitions = dataset.acquisitions[:]
        if acquisitions[-1].traj.shape[-1] > 0:
            print("Trajectory information found in ISMRMRD file.")
            kdata = KData.from_file(list_rawdata[0], KTrajectoryIsmrmrd())
        else:
            print(
                "No trajectory information found in ISMRMRD file, assume Cartesian scan."
            )
            kdata = KData.from_file(list_rawdata[0], KTrajectoryCartesian())

    csm = CsmData.from_kdata_inati(kdata[0])
    direct_reconstruction = DirectReconstruction(kdata=kdata, csm=csm)
    idata = direct_reconstruction(kdata)

    idata.to_dicom_folder(fpath_output / "IDATA")

    # T2 mapping if there are multiple echoes
    if len(idata.header.te) > 1:
        model = MonoExponentialDecay(decay_time=idata.header.te)
        dictionary = DictionaryMatchOp(model, index_of_scaling_parameter=0)
        dictionary.append(torch.tensor(1.0), torch.linspace(0.01, 0.4, 1000)[None, :])
        m0_match, t2_match = dictionary(idata.rss())
        t2_map = IData(data=t2_match.unsqueeze(0), header=idata[0].header)
        t2_map.to_dicom_folder(fpath_output / "T2MAP")

    return 0


path_in = Path(sys.argv[1])
path_out = Path(sys.argv[2])

if __name__ == "__main__":
    status = mr_direct_recon(path_in, path_out)
    sys.exit(status)
