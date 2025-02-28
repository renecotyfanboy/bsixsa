import numpy as np
import xspec
import os


truth = {
    1: 0.2,
    5: 0.2,
    10:1.05,
    11:3.1,
    14:2.5,
    15:1e-2
}

if __name__ == "__main__":

    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.AllModels.lmod("relxill")

    xspec.Xset.restore("model.xcm")

    xspec_model = xspec.AllModels(1)

    xspec_model.setPars(truth)


    settings = xspec.FakeitSettings(response="athena_xifu_4eV_gaussian.rmf", arf="athena_xifu_13_rows_no_filter.arf", exposure=5, fileName="spectrum_raw.pha")
    xspec.AllData.fakeit(settings=settings)
    os.system(f'ftgrouppha infile=spectrum_raw.pha outfile=spectrum_opt.pha grouptype=opt respfile=athena_xifu_4eV_gaussian.rmf')