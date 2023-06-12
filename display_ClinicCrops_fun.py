def display_ClinicCrops(plan_ct_nii,graph_coords,tumor,plan_ct_LM_nii,corrupt_index=None):
    #axial
    count=0
    for i in range(0,plan_ct_LM_nii.shape[0],5):
    #i = corrupt_index[0]
        if np.sum(plan_ct_LM_nii[i,:,:])>0:
            count+=1
            fig, ax = plt.subplots()
            ax.imshow(plan_ct_nii[i,:,:],'gray'),plt.axis('on')
            if tumor is not None: 
                ax.contour((tumor[i,:,:]),colors='green')
            ax.contour((plan_ct_LM_nii[i,:,:]),colors='yellow')
            plt.gca().add_patch(plt.Rectangle((graph_coords[2], graph_coords[1]),  graph_coords[5] - graph_coords[2],graph_coords[4] - graph_coords[1], fill=False, edgecolor='red'))
            plt.tight_layout()
            plt.text(10, 10, f"Slice: {i}", fontsize=10,color="white")
            plt.show()
            if count>4:
                break

    #coronal
    count=0
    for i in range(0,plan_ct_LM_nii.shape[1],5):
    #i = corrupt_index[1]
        if np.sum(plan_ct_LM_nii[:,i,:])>0:
            count+=1
            fig, ax = plt.subplots()
            ax.imshow(plan_ct_nii[:,i,:],'gray'),plt.axis('on')
            if tumor is not None: 
                ax.contour((tumor[:,i,:]),colors='green')
            ax.contour((plan_ct_LM_nii[:,i,:]),colors='yellow')
            plt.gca().add_patch(plt.Rectangle((graph_coords[2], graph_coords[0]), graph_coords[5] - graph_coords[2], graph_coords[3] - graph_coords[0], fill=False, edgecolor='red'))
            plt.tight_layout()
            plt.show()
            if count>4:
                break

    #sagital
    count=0
    for i in range(0,plan_ct_LM_nii.shape[2],5):
    #i = corrupt_index[2]
        if np.sum(plan_ct_LM_nii[:,:,i])>0:
            count+=1
            fig, ax = plt.subplots()
            ax.imshow(plan_ct_nii[:,:,i],'gray'),plt.axis('on')
            if tumor is not None: 
                ax.contour((tumor[:,:,i]),colors='green')
            ax.contour((plan_ct_LM_nii[:,:,i]),colors='yellow')
            plt.gca().add_patch(plt.Rectangle((graph_coords[1], graph_coords[0]),  graph_coords[4] - graph_coords[1],graph_coords[3] - graph_coords[0], fill=False, edgecolor='red'))
            plt.tight_layout()
            plt.show()
            if count>4:
                break

