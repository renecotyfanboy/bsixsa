import numpy as np
import xspec
import os
from astropy.io import fits

truth = {
    2: 4,
    3 :1.2,
    4: 0.02,
    5: 120,
    6: 1,
    7: 2,
    8: 1,
    9: 0.02,
    10: 100,
    11: 1.5,
}

def write_keywords_in_fits_file(fname, keywords_dict):
    """
    Write a dictionary of keywords to the FITS file header.

    Parameters:
    - fname (str): Path to the FITS file.
    - keywords_dict (dict): Dictionary where keys are FITS keywords and values are the values to write.
    """
    with fits.open(fname, mode='update') as hdul:  # Open in update mode to modify header
        header = hdul[1].header  # Access the header of the first extension (index 1)

        for keyword, value in keywords_dict.items():
            header[keyword] = value  # Write each keyword-value pair to the header

        # Save changes
        hdul.flush()  # Ensure changes are saved to the file


if __name__ == "__main__":
    """
    xspec.AllData.clear()
    xspec.AllModels.clear()
    
    bkg_settings = xspec.FakeitSettings(
        response="athena_xifu_4eV_gaussian.rmf",
        arf="athena_xifu_13_rows_no_filter.arf",
        exposure=15, fileName="bkg_raw_for_fakeit.pha"
    )

    xspec_model = xspec.Model("tbabs*powerlaw")
    xspec_model.setPars([0.1, 1., 1e-2])
    xspec.AllData.fakeit(settings=bkg_settings)

    print("Background spectrum counts 1 : ", np.sum(np.asarray(xspec.AllData(1).values)*xspec.AllData(1).exposure))

    xspec.AllData.clear()
    xspec.AllModels.clear()

    bkg_settings = xspec.FakeitSettings(
        response="athena_xifu_4eV_gaussian.rmf",
        arf="athena_xifu_13_rows_no_filter.arf",
        exposure=15, fileName="bkg_raw.pha"
    )

    xspec_model = xspec.Model("tbabs*powerlaw")
    xspec_model.setPars([0.1, 1., 1e-2])
    xspec.AllData.fakeit(settings=bkg_settings)

    print("Background spectrum counts 1 : ", np.sum(np.asarray(xspec.AllData(1).values)*xspec.AllData(1).exposure))
    """

    xspec.AllData.clear()
    xspec.AllModels.clear()

    xspec.Xset.restore("model.xcm")
    xspec_model = xspec.AllModels(1)
    xspec_model.setPars(truth)

    obs_settings = xspec.FakeitSettings(
        response="athena_xifu_4eV_gaussian.rmf",
        arf="athena_xifu_13_rows_no_filter.arf",
        background="bkg_raw_for_fakeit.pha",
        exposure=50, fileName="spectrum_raw.pha"
    )

    xspec.AllData.fakeit(settings=obs_settings)

    #write_keywords_in_fits_file("spectrum_raw.pha", {"BACKFILE" :"bkg_raw.pha"})

    os.system(f'ftgrouppha infile=spectrum_raw.pha outfile=spectrum_opt.pha grouptype=opt respfile=athena_xifu_4eV_gaussian.rmf')

    print("Source spectrum counts: ", np.sum(np.asarray(xspec.AllData(1).values)*xspec.AllData(1).exposure))
