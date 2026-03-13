import numpy as np


def smear_constituents(constituents, eta_smear_factor=0.15, phi_smear_factor=0.15):
    smeared = constituents.copy()

    my_eta = smeared[:, 0]
    my_phi = smeared[: , 1]

    eta_sigma_num = eta_smear_factor * 2.5 #is -2.5 to +2.5 the right range for this value?
    phi_sigma_num = phi_smear_factor * (2 * np.pi)

    smeared[:, 0] = my_eta + np.random.normal(0, eta_sigma_num, my_eta.shape)
    smeared[:, 1] = my_phi + np.random.normal(0, phi_sigma_num, my_phi.shape)

    return smeared


def smear_dataset(data, eta_smear_factor=0.15, phi_smear_factor = 0.15):
    smeared_data = data.copy()

    for i in range(len(data)):
        smeared_data[i] = smear_constituents(data[i], eta_smear_factor=eta_smear_factor,
                                              phi_smear_factor=phi_smear_factor)
    
    return smeared_data



