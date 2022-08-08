dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
	'ioct_train_segmentation': 'data/ioct/labels/train',
	'ioct_train': 'data/ioct/bscans/train',
	'ioct_test_segmentation': 'data/ioct/labels/test',
	'ioct_test': 'data/ioct/bscans/test',
	'ioct_overfit_segmentation': 'data/overfit/labels/train',
	'ioct_overfit': 'data/overfit/bscans/train'
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar',
	'celebs_seg2face': 'pretrained_models/psp_celebs_seg_to_face.pt',
}
