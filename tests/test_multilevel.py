import os
import pickle
import unittest
import tempfile

import formulaic
import numpy
import pandas

import slimp

class TestMultiLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "data", "sleepstudy.csv"))
        cls.data["Days"] = cls.data["Days"].astype(float)
        cls.data["Subject"] = cls.data["Subject"].astype("category")

        cls.formula = ["Reaction ~ 1+Days", ("Subject", "1+Days")]
        
        cls.outcomes, unmodeled_predictors = formulaic.model_matrix(
            cls.formula[0], cls.data)
        
        modeled_predictors = formulaic.model_matrix(cls.formula[1][1], cls.data)
        modeled_predictors.index = cls.data[cls.formula[1][0]]
        cls.predictors = [unmodeled_predictors, modeled_predictors]
    
    def test_no_sample(self):
        def dump(path):
            model = slimp.Model(self.formula, self.data, seed=42, num_chains=4)
            with open(path, "wb") as fd:
                pickle.dump(model, fd)
        def load(path):
            with open(path, "rb") as fd:
                return pickle.load(fd)
        
        with tempfile.TemporaryDirectory() as dir:
            dump(os.path.join(dir, "model.pkl"))
            model = load(os.path.join(dir, "model.pkl"))
            
            self._test_model_data(model)
            self._test_sampler_parameters(model)
            self.assertTrue(model.draws is None)
    
    def test_sample(self):
        def dump(path):
            model = slimp.Model(self.formula, self.data, seed=42, num_chains=4)
            model.sample()
            with open(path, "wb") as fd:
                pickle.dump(model, fd)
        def load(path):
            with open(path, "rb") as fd:
                return pickle.load(fd)
        
        with tempfile.TemporaryDirectory() as dir:
            dump(os.path.join(dir, "model.pkl"))
            model = load(os.path.join(dir, "model.pkl"))
            
            self._test_model_data(model)
            self._test_sampler_parameters(model)
            self._test_model_diagnostics(model)
            self._test_model_draws(model)
            self._test_model_prior_predict(model)
            self._test_model_posterior_epred(model)
            self._test_model_posterior_predict(model)
            self._test_r_squared(model)
    
    def _test_model_data(self, model):
        self.assertEqual(self.formula, model.formula)
        self.assertTrue(self.data.equals(model.data))
        
        for p1, p2 in zip(self.predictors, model.predictors):
            self.assertTrue(p1.equals(p2))
        self.assertTrue(self.outcomes.equals(model.outcomes))
    
    def _test_sampler_parameters(self, model):
        self.assertEqual(model.sampler_parameters.seed, 42)
        self.assertEqual(model.sampler_parameters.num_chains, 4)
        self.assertEqual(model.sampler_parameters.num_samples, 1000)
    
    def _test_model_diagnostics(self, model):
        numpy.testing.assert_equal(
            model.hmc_diagnostics.max().values[:2], [0, 0])
        numpy.testing.assert_allclose(
            model.hmc_diagnostics["e_bfmi"].values,
            [0.82662801, 0.84384758, 0.79167552, 0.82511096])
        
        summary = model.summary()
        numpy.testing.assert_allclose(
            summary["R_hat"].values,
            [
                1.0022431791188962, 1.0010054014196024, 1.0022777333813615,
                0.9997558030114256, 1.0001979694720926, 1.00069307056117,
                0.9996528887427174, 1.0002676183887742, 0.9997543751129961,
                0.9995250151536081, 0.9995989541362765, 0.999542242920801,
                0.9997037141097187, 0.9997904744539101, 0.9997717801580536,
                1.000309464797351, 0.9996217325495173, 0.9995949952330482,
                1.0008630140154104, 1.0002282498192405, 0.9998947700866868,
                1.0004523967328214, 1.0012889524316957, 1.0013772946724095,
                1.0015310684903311, 0.9995312276229742, 1.0000944086681867,
                1.0007783854336814, 1.0001251999370613, 1.000352792408075,
                1.0000902124891986, 1.000158279215234, 1.0013474503791684,
                1.000425701616859, 0.9999535702016961, 1.0000753020364859,
                1.001841415607893, 1.0007851220640214, 1.0003874072046497,
                1.0008639053121937, 0.9999171672993182, 0.9998223798270969,
                numpy.nan, 1.0000976768507024, numpy.nan, 0.9996257059257454,
                0.9996548802270268, 1.0007797800686358, 1.0007797800686358,
                0.9998521902392861, 0.99995976525385])
        numpy.testing.assert_allclose(
            summary["N_Eff"].values,
            [
                4126.65137537128, 3961.209859576136, 2662.111315508689,
                3686.117516891521, 3935.4650232801027, 3851.9910209177497,
                3927.41468241427, 3774.5190668613836, 4057.1810987182685,
                3778.1459234855793, 3766.306961721788, 3919.069645851634,
                4134.2954080782565, 3932.9156006969806, 4061.0770344701677,
                3888.9680640472984, 3913.955391539713, 3892.365275532672,
                3956.9449137561846, 4015.308018208126, 3689.5353270764067,
                3945.2594721124865, 3960.4297092512575, 3954.1682254313505,
                4082.2627614941784, 3909.496738859146, 3929.9761605991002,
                3885.8058064033557, 3977.1990633165456, 3672.7561300001844,
                4112.89956451621, 3923.520096947475, 4010.161383846983,
                3930.167748873263, 4146.126127730415, 4203.982278585879,
                3960.4842446900516, 3858.080965452658, 4108.326578065958,
                3967.1090445337604, 3724.7637534261175, 4036.3079777925054,
                numpy.nan, 4077.489471957915, numpy.nan, 4004.2036646084107,
                3775.0466079394087, 4097.73281581796, 4097.73281581796,
                4036.3138402434074, 3736.7546758381236])
    
    def _test_model_draws(self, model):
        self.assertEqual(
            list(model.draws.columns),
            [
                "Intercept_c", "Days", "sigma_y", "Subject[308]/Intercept",
                "Subject[309]/Intercept", "Subject[310]/Intercept",
                "Subject[330]/Intercept", "Subject[331]/Intercept",
                "Subject[332]/Intercept", "Subject[333]/Intercept",
                "Subject[334]/Intercept", "Subject[335]/Intercept",
                "Subject[337]/Intercept", "Subject[349]/Intercept",
                "Subject[350]/Intercept", "Subject[351]/Intercept",
                "Subject[352]/Intercept", "Subject[369]/Intercept",
                "Subject[370]/Intercept", "Subject[371]/Intercept",
                "Subject[372]/Intercept", "Subject[308]/Days",
                "Subject[309]/Days", "Subject[310]/Days", "Subject[330]/Days",
                "Subject[331]/Days", "Subject[332]/Days", "Subject[333]/Days",
                "Subject[334]/Days", "Subject[335]/Days", "Subject[337]/Days",
                "Subject[349]/Days", "Subject[350]/Days", "Subject[351]/Days",
                "Subject[352]/Days", "Subject[369]/Days", "Subject[370]/Days",
                "Subject[371]/Days", "Subject[372]/Days", "sigma_Beta.1",
                "sigma_Beta.2", "L_Omega_Beta.1.1", "L_Omega_Beta.2.1",
                "L_Omega_Beta.1.2", "L_Omega_Beta.2.2",
                "Sigma_Beta[Intercept, Intercept]",
                "Sigma_Beta[Days, Intercept]", "Sigma_Beta[Intercept, Days]",
                "Sigma_Beta[Days, Days]", "Intercept"])
        
        numpy.testing.assert_allclose(
            model.draws.iloc[:5,:5].values,
            [
                [
                    291.7380465072354, 10.980317859639777, 23.832518251905064,
                    29.01726605946373, -32.66177755392848],
                [
                    286.38116863050965, 10.28145484597827, 23.740875872976236,
                    2.6894964952267593, -27.86971974112354],
                [
                    311.97974531637794, 13.53749830458907, 27.325004047132442,
                    1.463942989881874, -36.286337934179336],
                [
                    286.3077712012768, 9.034580521801457, 26.481399786709463,
                    1.5439511653106794, -25.13955693839785],
                [
                    309.61499750684493, 10.382281794995833, 26.430691433538055,
                    -30.96794700917114, -55.56368214437405]])
    
    def _test_model_prior_predict(self, model):
        self.assertEqual(
            model.prior_predict.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.prior_predict.iloc[:5,:5].values,
            [
                [
                    168.57826547863624, 128.89129713405902, 114.55161341940249,
                    162.00761394985278, 78.27418462598445],
                [
                    675.2758686029731, 631.8258650115947, 603.2770169483491,
                    581.9840929319839, 566.1330243066197],
                [
                    903.1230247502729, 762.3355479810986, 670.654246851972,
                    632.453361458342, 534.4377613195647],
                [
                    -609.3152153069562, -404.89592880710455, -120.709994672792,
                    184.1835146636016, 409.60479930319576],
                [
                    -570.8783902748223, -466.4240766628323, -358.9375039263816,
                    -250.15617357346446, -89.86734659390068]])
    
    def _test_model_posterior_epred(self, model):
        self.assertEqual(
            model.posterior_epred.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_epred.iloc[:5,:5].values,
            [
                [
                    271.34388219832016, 288.2469007462204, 305.14991929412054,
                    322.0529378420207, 338.9559563899209],
                [
                    238.12941268265104, 259.5226895483043, 280.9159664139576,
                    302.3092432796109, 323.70252014526415],
                [
                    250.27435774954282, 271.67502243696663, 293.07568712439036,
                    314.47635181181414, 335.8770164992379],
                [
                    279.84259755828134, 296.2880014073931, 312.73340525650474,
                    329.1788091056165, 345.6242129547282],
                [
                    242.80411831883418, 264.76423562735124, 286.7243529358682,
                    308.68447024438524, 330.6445875529023]])
    
    def _test_model_posterior_predict(self, model):
        self.assertEqual(
            model.posterior_predict.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_predict.iloc[:5,:5].values,
            [
                [
                    256.50440043606955, 295.21660573194464, 325.99301710858686,
                    366.3406174915792, 352.1404400084677],
                [
                    287.60645962339566, 240.07063329765563, 399.3833930923654,
                    368.41280878700354, 439.78117019709333],
                [
                    288.07649461246524, 307.23063744463076, 341.1974370918123,
                    348.89134266219423, 376.41493704481167],
                [
                    316.85642335932783, 379.2571567186782, 307.17502241144183,
                    382.68021962455435, 400.32198703815584],
                [
                    215.86602827492337, 221.69407371258106, 312.7467421285552,
                    321.6817626008916, 378.1577361636175]])
    
    def _test_r_squared(self, model):
        r_squared = slimp.stats.r_squared(model)
        self.assertEqual(r_squared.shape, (4000,))
        numpy.testing.assert_allclose(
            r_squared.iloc[:25],
            [
                0.8248120146848996, 0.8177906588810865, 0.7781337302892255,
                0.8010484005805698, 0.7776708119454869, 0.7846283452600643,
                0.8222930215617418, 0.8054522035953261, 0.8127278443755166,
                0.8220138365768451, 0.8274894433055439, 0.7973495609588798,
                0.7886787353337005, 0.7798119123195185, 0.8307643215394238,
                0.8082905990940911, 0.7856625373132731, 0.7554314696625409,
                0.7684411753180886, 0.7692149578342548, 0.7874648481957183,
                0.7918708122871003, 0.7626919795563907, 0.8026301725034937,
                0.7707909064832619])
    
if __name__ == "__main__":
    unittest.main()
