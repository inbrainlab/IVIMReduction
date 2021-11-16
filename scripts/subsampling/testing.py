###########################################################
#                                                         #
#                   Testing Script                        #
#                                                         #
##########################################################
from subsampling import test_subsampling, test_bvals_selection_image

if __name__ == "__main__":
    test_subsampling()
    test_bvals_selection_image()
    print("Tests passed")
