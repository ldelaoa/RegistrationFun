import SimpleITK as sitk
import numpy as np
import os
import math


def RotationScale(transform):
    parameters = transform.GetParameters()
    rotationX = parameters[0]
    rotationY = parameters[1]
    rotationZ = parameters[2]
    rotationXDegrees = rotationX * 180.0 / math.pi
    rotationYDegrees = rotationY * 180.0 / math.pi
    rotationZDegrees = rotationZ * 180.0 / math.pi
    rotationList = [rotationXDegrees, rotationYDegrees, rotationZDegrees]

    components = transform.GetNthTransform(0).GetMatrix()
    matrix = np.array(components).reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    real_eigenvalues = np.real(eigenvectors)

    return real_eigenvalues[0], real_eigenvalues[1], real_eigenvalues[2], rotationList[0], rotationList[1],rotationList[2]


def Register_fun_v4(planning_ct_np,lowdose_ct_np,pet_np,patient_number):
    #SPECS:
    #VersorRigid3DTransform,Geomerty, MattesMutualInfo, RegularStepGradientDescent

    # Load the planning CT and low-dose CT
    planning_ct = sitk.GetImageFromArray(planning_ct_np)
    lowdose_ct = sitk.GetImageFromArray(lowdose_ct_np)

    # Register the low-dose CT to the planning CT
    fixed_image = sitk.Cast(planning_ct, sitk.sitkFloat32)
    moving_image = sitk.Cast(lowdose_ct, sitk.sitkFloat32)

    #INITIALIZE
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.VersorRigid3DTransform(),sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    #METRICS
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    #OPTIMIZER
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, numberOfIterations=500,minStep=0.0001,gradientMagnitudeTolerance=1e-8)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    #Registration
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    evaluationMetric = registration_method.GetMetricValue()
    #print(f"Final metric value for patient {patient_number}: {evaluationMetric}")
    #print(f"Optimizer's stopping condition for patient {patient_number}: {registration_method.GetOptimizerStopConditionDescription()}")
    #print(f"Iteration for patient {patient_number}: {registration_method.GetOptimizerIteration()}")
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

    sX,sY,sZ,rX,rY,rZ = RotationScale(final_transform)

    return ldct_registered_np,pet_registered_np,evaluationMetric,sX,sY,sZ,rX,rY,rZ