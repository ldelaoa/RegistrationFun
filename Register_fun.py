import SimpleITK as sitk
import numpy as np
import os


def Register_fun(planning_ct_np,lowdose_ct_np,pet_np,patient_number):

    # Load the planning CT and low-dose CT
    planning_ct = sitk.GetImageFromArray(planning_ct_np)
    lowdose_ct = sitk.GetImageFromArray(lowdose_ct_np)

    # Register the low-dose CT to the planning CT
    fixed_image = sitk.Cast(planning_ct, sitk.sitkFloat32)
    moving_image = sitk.Cast(lowdose_ct, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=400, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False) 
    
    final_transform = registration_method.Execute(fixed_image, moving_image) 
    
    print(f"Final metric value for patient {patient_number}: {registration_method.GetMetricValue()}") #MattesSimilarityMetric is similar to NCC, both use correlation in intensity
    print(f"Optimizer's stopping condition for patient {patient_number}: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Iteration for patient {patient_number}: {registration_method.GetOptimizerIteration()}")
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Save the registered low-dose CT and PET
    if False:
        registered_lowdose_ct_path = os.path.join(output_dir, f"CT_image_REG_{patient_number}.nii.gz")
        registered_pet_path = os.path.join(output_dir, f"PET_image_REG_{patient_number}.nii.gz")
        sitk.WriteImage(moving_resampled, registered_lowdose_ct_path)

    # Load and register the PET image
    pet_image = sitk.GetImageFromArray(pet_np)
    registered_pet_image = sitk.Resample(pet_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    if False:
        sitk.WriteImage(registered_pet_image, registered_pet_path)
        print(f'PET for patient {patient_number} registrated and saved.')
    
    ldct_registered_np = sitk.GetArrayFromImage(moving_resampled)
    pet_registered_np = sitk.GetArrayFromImage(registered_pet_image)
    
    return ldct_registered_np,pet_registered_np
    
