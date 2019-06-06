import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path

# COCO bounding boxes are 0-indexed

class RETAIL(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(RETAIL, self).__init__(db_config)

        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._coco_cls_ids = [
		1,2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,58,59,60,
		61,62,63,64,65,66,67,68,69,70,
		71,72,73,74,75,76,77,78,79,80,
		81,82,83,84,85,86,87,88,89,90,
		91,92,93,94,95,96,97,98,99,100,
		101,102,103,104,105,106,107,108,109,110,
		111,112,113,114,115,116,117,118,119,120,
		121,122,123,124,125,126,127,128,129,130,
		131,132,133,134,135,136,137,138,139,140,
		141,142,143,144,145,146,147,148,149,150,
		151,152,153,154,155,156,157,158,159,160,
		161,162,163,164,165,166,167,168,169,170,
		171,172,173,174,175,176,177,178,179,180,
		181,182,183,184,185,186,187,188,189,190,
		191,192,193,194,195,196,197,198,199,200
        ]

        self._coco_cls_names = [
		"shanghaojiahelandou55g","caiyuanxiaobing80g","shanghaojiaxianxiapian40g","shanghaojiaxieweiyizu40g","miaocuijiaomolitanshaowei65g",
		"panpanshaokaoniupaiweikuai105g","shanghaojiaxianxiatiao40g","shanghaojiayangcongquan40g","shanghaojiarishiyuguohaitaiwei50g","qiduorishiniupaiwei90g",
		"qiduomeishihuojiwei90g","shanghaojiasumitiaocaomeiwei40g","ganyuanxiehuangweiguaziren75g","huiyikaixinguo140g","huiyixianweihuasheng350g",
		"huiyiyaoguo160g","huiyijuqi100g","huiyidiguagan228g","huiyitaiguomangguogan80g","huiyihuangtaoguogan75g",
		"huiyininmengpian65g","xinjianghetiantanzao454g","huiyixianggu100g","huiyiguiyuangan500g","huiyichashugu200g",
		"haoxiongdanpianheimuer150g","huiyizhuhuasheng454g","huiyihuanghuacai100g","qiaqialiangchaguazi150g","qiaqianaixiangweiguazi150g",
		"chezichabaolvcha50g","chezichabaohongcha50g","youlemeixiangyuwei80g","youlemeihongdounaicha65g","huannichongdiaotudouzhou25g",
		"jiangzhonghouguzaocanmixi40g","yonghedoujiangtiandoujiangfen210g","lidunninmengfengweicha180g","guigeduozhongmeiguomaipian40g","rongyigumaijiaheimiwei30g",
		"rongyigumaijiahongdouwei30g","jinyexianglaniuroumian112g","jinyelaotansuancainiuroumian118g","jinyehongshaoniuroumian114g","heweidaohaixianfengwei84g",
		"kangshifubaihujiaorougumian76g","kangshifuxianglaniuroumian105g","kangshifuxianglasuanweipaigumian108g","kangshifutengjiaoniuroumian82g","huafengjirousanxianyimian87g",
		"kangshifuheihujiaoniupaimian104g","wugudaochanghongshaoniuroumian100g","kangshifulaotansuancainiuroumian114g","ajipaofubingganmangguoboluowei60g","qinglianlanmeiweijiaxinbing63g",
		"qinglianfengliweijiaxinbing63g","qingliancaomeiweijiaxinbing63g","jiadunweihuabinggancaomeiwei50g","jiadunweihuabingganninmengwei50g","aishilexiangcaoniunaiwei50g",
		"aishileqiaokeliwei50g","bailizihaitaiwei60g","bailizicaomeiniunaiwei45g","quechaocuicuisha80g","nabaodiqiaokeliweiweihua58g",
		"guilidizhonghaifengweimianbaotiao50g","kangshifumiaofuqiaokeliwei48g","aixiangqinchangpianmianbao90g","daliyuanpaicaomeiweidangezhuang*","miniaoliao55g",
		"nongfushanquankuangquanshui550ml","yibaokuangquanshui555ml","kekoukelelingdu500ml","kekoukele500ml","baishikele600ml",
		"fendapingguowei500ml","fendachengwei500ml","xuebi500ml","xilipijiu500ml","baiweipijiu600ml",
		"baishikele330ml","kekoukele330ml","wanglaoji310ml","chapaiyouzilvcha500ml","chapaimeiguilizhihongcha500ml",
		"kangshifubinghongcha250ml","jiaduobao250ml","rioguojiushuimitaowei275ml","rioguojiulanmeiguiweishijiwei275ml","niulanshanerguotou100ml",
		"haerbinpijiu330ml","qingdaopijiu330ml","xuehuapijiu330ml","haerbinpijiu500ml","kelerpijiu500ml",
		"baiweipijiu500ml","qqxingquancongnai125ml","qqxingjunshannai125ml","wahahaadgainai220g","huolibaodongliyuan105ml",
		"wangziniunaifuyuanru250ml","yilichunniunai250ml","weitaditangyuanweidounai250ml","baiyihuashengniunai250ml","huiyiyuanweidounai250ml",
		"yiliyousuanru250ml","yilizaocannai250ml","daliyuanguiyuanlianzi360g","yinlubingtangbaiheyiner280g","xiduoduoshijinyeguo567g",
		"duleboluokuai567g","duleboluokuai234g","yinluyirenhongdouzhou280g","yinlulianziyumizhou280g","yinluzishuzimizhou280g",
		"yinluyenaiyanmaizhou280g","yinluheitangguiyuan280g","meilinwucanrou340g","zhujiangqiaopaidouchiyu150g","gulongyuanweihuanghuayu120g",
		"xiongjibiaoyejiang140ml","defumangguosuannaiqiaokeli42g","defumokabadanmuqiaokeli43g","defubaixiangguobaiqiaokeli42g","mmhuashengniunaiqiaokelidou40g",
		"mmniunaiqiaokelidou40g","haoshiniunaiqiaokeli40g","haoshiquqinaixiangbaiqiaokeli40g","cuixiangmihaitaibaiqiaokeli24g","cuixiangminaixiangbaiqiaokeli24g",
		"shilijiahuashengjiaxinqiaokeli51g","shilijiayanmaihuashengjiaxinqiaokeli40g","shilijialahuashengjiaxinqiaokeli40g","xuanmaiguoweilangbohewei37g","xuanmaiguoweilangninmengwei37g",
		"xuanmaibohewei21g","xuanmaiputaowei21g","xuanmaixiguawei21g","xuanmaiputaowei50g","lvjianwutangbohetangmolihuachawei34g",
		"lvjian5pianzhuang15g","bibabumianhuapaopaotangkelewei11g","bibabumianhuapaopaotangputaowei11g","xingbaobinfenyuanguowei25g","aerbeisijiaoxiangniunaiweiyingtang45g",
		"aerbeisiniunairuantanghuangtaosuannaiwei47g","aerbeisiniunairuantanglanmeisuannaiwei47g","wanglaojirunhoutang28g","yiliniunaipianlanmeiwei32g","xiongboshikoujiaotangcaomeiniunaiwei52g",
		"caihongtangyuanguowei45g","baodingtianyuchenniangmicu245ml","hengshunxiangcu340ml","taitailejijing200g","jialexianggujirongtangliao41g",
		"huiyilajiaofen15g","huiyishengjiangfen15g","weihaomeijiaoyan20g","haixingjiadianjingzhiyan400g","hengshunliaojiu500ml",
		"dongguweijixianjiangyou150ml","dongguyipinxianjiangyou150ml","xinheliuyuexianjiangyou160ml","lishidelinlingdushukoushui80ml","shufujiachunbaiqingxiangmuyulu100ml",
		"meitaodingxingzelishui60ml","qingyangnanshixifaluhuoliyundongbohexing50ml","lanyueliangfengqingbailanxiyiye80g","gaolujieliangbaixiaosuda180g","gaolujiebingshuang180g",
		"shulianghaochibai80g","yunnanbaiyaoyagao45g","shukebaobeiertongyashua","qingfengyuanmuchunpinjinzhuang100x3","jierouface150x3",
		"banbu100x3","weidayinger150x3","xiangyinxiaohuangren150x3","qingfengyuanmuchunpinheiyao150x3","jieyunrongchugan130x3",
		"shujiemengyinhua120x2","xiangyinhongyue130x3","debaopingguomuwei90x4","qingfengxinrenchunpin130x3","jinyuzhujianglv135x3",
		"qingfengyuanmuchunpin150x2","jierouface130x3","weidalitimei110x3","jieroucsdanbao*","xiangyinxiaohuangrendanbao*",
		"qingfengyuansedanbao*","xiangyinchayudanbao*","qingfengzhiganchunpindanbao*","miqi1928bijiben","guangbogutijiao15g",
		"piaojuwenjiandai","chenguangwoniugaizhengdai","hongtaiyetijiao50g","mapeidezizhanxingbiaoqian","dongyajihaobi",
        ]

        self._cls2coco  = {ind + 1: coco_id for ind, coco_id in enumerate(self._coco_cls_ids)}
        self._coco2cls  = {coco_id: cls_id for cls_id, coco_id in self._cls2coco.items()}
        self._coco2name = {cls_id: cls_name for cls_id, cls_name in zip(self._coco_cls_ids, self._coco_cls_names)}
        self._name2coco = {cls_name: cls_id for cls_name, cls_id in self._coco2name.items()}

        if split is not None:
            coco_dir = os.path.join(sys_config.data_dir, "retail")

            self._split     = {
                "trainval": "train2019",
                "minival":  "val2019",
                "testdev":  "test2019"
            }[split]
            self._data_dir  = os.path.join(coco_dir, "images", self._split)
            self._anno_file = os.path.join(coco_dir, "annotations", "instances_{}.json".format(self._split))

            self._detections, self._eval_ids = self._load_coco_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds   = np.arange(len(self._image_ids))

    def _load_coco_annos(self):
        from pycocotools.coco import COCO

        coco = COCO(self._anno_file)
        self._coco = coco

        class_ids = coco.getCatIds()
        image_ids = coco.getImgIds()
        
        eval_ids   = {}
        detections = {}
        for image_id in image_ids:
            image = coco.loadImgs(image_id)[0]
            dets  = []
            
            eval_ids[image["file_name"]] = image_id
            for class_id in class_ids:
                annotation_ids = coco.getAnnIds(imgIds=image["id"], catIds=class_id)
                annotations    = coco.loadAnns(annotation_ids)
                category       = self._coco2cls[class_id]
                for annotation in annotations:
                    det     = annotation["bbox"] + [category]
                    det[2] += det[0]
                    det[3] += det[1]
                    dets.append(det)

            file_name = image["file_name"]
            if len(dets) == 0:
                detections[file_name] = np.zeros((0, 5), dtype=np.float32)
            else:
                detections[file_name] = np.array(dets, dtype=np.float32)
        return detections, eval_ids

    def image_path(self, ind):
        if self._data_dir is None:
            raise ValueError("Data directory is not set")

        db_ind    = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return os.path.join(self._data_dir, file_name)

    def detections(self, ind):
        db_ind    = self._db_inds[ind]
        file_name = self._image_ids[db_ind]
        return self._detections[file_name].copy()

    def cls2name(self, cls):
        coco = self._cls2coco[cls]
        return self._coco2name[coco]

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2coco[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        from pycocotools.cocoeval import COCOeval

        if self._split == "testdev":
            return None

        coco = self._coco

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids  = [self._cls2coco[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]
