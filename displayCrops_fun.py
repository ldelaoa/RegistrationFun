def displayCrops(plan_ct_cropped_1,plan_ct_cropped_2,plan_ct_cropped_3):
    print("Crop1")
    for i in range(0,plan_ct_cropped_1.shape[0],10):
        plt.subplot(1,1,1),plt.imshow(plan_ct_cropped_1[i,:,:]),plt.axis('off')
#    plt.subplot(2,3,2),plt.imshow(plan_ct_cropped_1[i//2,:,:])
        plt.show()
    print("Crop2")
    for i in range(0,plan_ct_cropped_2.shape[0],10):
        plt.subplot(1,1,1),plt.imshow(plan_ct_cropped_2[i,:,:]),plt.axis('off')
 #   plt.subplot(2,3,4),plt.imshow(plan_ct_cropped_2[i//2,:,:])
        plt.show()
    print("Crop3")
    for i in range(0,plan_ct_cropped_3.shape[0],10):
        plt.subplot(1,1,1),plt.imshow(plan_ct_cropped_3[i,:,:]),plt.axis('off')
  #  plt.subplot(2,3,6),plt.imshow(plan_ct_cropped_3[i//2,:,:])
    
        plt.show()
    return 
