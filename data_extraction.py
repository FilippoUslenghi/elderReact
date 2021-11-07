"""
Questo script esegue l'estrazione dei dati sia con OpenFace che con MediaPipe.
"""

import os
import cv2
import mediapipe as mp
import pandas as pd


# Estrae i landmark e le action unit dei video del dataset con OpenFace
datasets = ['train', 'dev', 'test']

for dataset in datasets:
    # path dell'output di OpenFace
    out_dir = os.path.join('/', 'home', 'filippo', 'elderReact', 'openFace', dataset, 'processed')
    os.makedirs(out_dir, exist_ok=True)

    for video in os.listdir(os.path.join('dataset','ElderReact_Data', 'ElderReact_' + dataset)):
        videoName = video[:-4]
        if f'{videoName}.csv' in os.listdir(out_dir): continue # skip the already processed videos in case you have to stop the script
        os.system(f'./FeatureExtraction -f {video} -2Dfp -aus -out_dir {out_dir}')
        # print(f'./FeaturesExtraction -f {video} -2Dfp -aus -out_dir {out_dir}')
os.system(f'rm {out_dir}/*.txt')

# Estrae i landmark con MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

for dataset in datasets:
    base_dir = os.path.join('dataset','ElderReact_Data',f'ElderReact_{dataset}','')
    out_dir = os.path.join('mediaPipe', dataset, 'processed')
    os.makedirs(out_dir, exist_ok=True)

    for video in os.listdir(base_dir):
        videoName = video[:-4]
        if f'{videoName}.csv' in os.listdir(out_dir): continue # skip the already processed videos in case you ahve to stop the script
        cap = cv2.VideoCapture(os.path.join(base_dir, video))
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5
        video_df = pd.DataFrame()
        with mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence) as face_mesh:
            frame = 0
            while cap.isOpened():
                success, image = cap.read()
                if frame < cap.get(cv2.CAP_PROP_FRAME_COUNT): frame+=1
                if not success: break

                # Convert the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_mesh.process(image)
                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    landmark_x=[int(landmark.x*image.shape[1]) for landmark in results.multi_face_landmarks[0].landmark]
                    landmark_y=[int(landmark.y*image.shape[0]) for landmark in results.multi_face_landmarks[0].landmark]

                    landmarks_df = pd.DataFrame({'frame':frame, 'min_detection_cf': min_detection_confidence, 'min_tracking_cf': min_tracking_confidence, 'x_0':landmark_x[0], 'x_1':landmark_x[1], 'x_2':landmark_x[2], 'x_3':landmark_x[3], 'x_4':landmark_x[4], 'x_5':landmark_x[5], 'x_6':landmark_x[6], 'x_7':landmark_x[7], 'x_8':landmark_x[8], 'x_9':landmark_x[9], 'x_10':landmark_x[10], 'x_11':landmark_x[11], 'x_12':landmark_x[12], 'x_13':landmark_x[13], 'x_14':landmark_x[14], 'x_15':landmark_x[15], 'x_16':landmark_x[16], 'x_17':landmark_x[17], 'x_18':landmark_x[18], 'x_19':landmark_x[19], 'x_20':landmark_x[20], 'x_21':landmark_x[21], 'x_22':landmark_x[22], 'x_23':landmark_x[23], 'x_24':landmark_x[24], 'x_25':landmark_x[25], 'x_26':landmark_x[26], 'x_27':landmark_x[27], 'x_28':landmark_x[28], 'x_29':landmark_x[29], 'x_30':landmark_x[30], 'x_31':landmark_x[31], 'x_32':landmark_x[32], 'x_33':landmark_x[33], 'x_34':landmark_x[34], 'x_35':landmark_x[35], 'x_36':landmark_x[36], 'x_37':landmark_x[37], 'x_38':landmark_x[38], 'x_39':landmark_x[39], 'x_40':landmark_x[40], 'x_41':landmark_x[41], 'x_42':landmark_x[42], 'x_43':landmark_x[43], 'x_44':landmark_x[44], 'x_45':landmark_x[45], 'x_46':landmark_x[46], 'x_47':landmark_x[47], 'x_48':landmark_x[48], 'x_49':landmark_x[49], 'x_50':landmark_x[50], 'x_51':landmark_x[51], 'x_52':landmark_x[52], 'x_53':landmark_x[53], 'x_54':landmark_x[54], 'x_55':landmark_x[55], 'x_56':landmark_x[56], 'x_57':landmark_x[57], 'x_58':landmark_x[58], 'x_59':landmark_x[59], 'x_60':landmark_x[60], 'x_61':landmark_x[61], 'x_62':landmark_x[62], 'x_63':landmark_x[63], 'x_64':landmark_x[64], 'x_65':landmark_x[65], 'x_66':landmark_x[66], 'x_67':landmark_x[67], 'x_68':landmark_x[68], 'x_69':landmark_x[69], 'x_70':landmark_x[70], 'x_71':landmark_x[71], 'x_72':landmark_x[72], 'x_73':landmark_x[73], 'x_74':landmark_x[74], 'x_75':landmark_x[75], 'x_76':landmark_x[76], 'x_77':landmark_x[77], 'x_78':landmark_x[78], 'x_79':landmark_x[79], 'x_80':landmark_x[80], 'x_81':landmark_x[81], 'x_82':landmark_x[82], 'x_83':landmark_x[83], 'x_84':landmark_x[84], 'x_85':landmark_x[85], 'x_86':landmark_x[86], 'x_87':landmark_x[87], 'x_88':landmark_x[88], 'x_89':landmark_x[89], 'x_90':landmark_x[90], 'x_91':landmark_x[91], 'x_92':landmark_x[92], 'x_93':landmark_x[93], 'x_94':landmark_x[94], 'x_95':landmark_x[95], 'x_96':landmark_x[96], 'x_97':landmark_x[97], 'x_98':landmark_x[98], 'x_99':landmark_x[99], 'x_100':landmark_x[100], 'x_101':landmark_x[101], 'x_102':landmark_x[102], 'x_103':landmark_x[103], 'x_104':landmark_x[104], 'x_105':landmark_x[105], 'x_106':landmark_x[106], 'x_107':landmark_x[107], 'x_108':landmark_x[108], 'x_109':landmark_x[109], 'x_110':landmark_x[110], 'x_111':landmark_x[111], 'x_112':landmark_x[112], 'x_113':landmark_x[113], 'x_114':landmark_x[114], 'x_115':landmark_x[115], 'x_116':landmark_x[116], 'x_117':landmark_x[117], 'x_118':landmark_x[118], 'x_119':landmark_x[119], 'x_120':landmark_x[120], 'x_121':landmark_x[121], 'x_122':landmark_x[122], 'x_123':landmark_x[123], 'x_124':landmark_x[124], 'x_125':landmark_x[125], 'x_126':landmark_x[126], 'x_127':landmark_x[127], 'x_128':landmark_x[128], 'x_129':landmark_x[129], 'x_130':landmark_x[130], 'x_131':landmark_x[131], 'x_132':landmark_x[132], 'x_133':landmark_x[133], 'x_134':landmark_x[134], 'x_135':landmark_x[135], 'x_136':landmark_x[136], 'x_137':landmark_x[137], 'x_138':landmark_x[138], 'x_139':landmark_x[139], 'x_140':landmark_x[140], 'x_141':landmark_x[141], 'x_142':landmark_x[142], 'x_143':landmark_x[143], 'x_144':landmark_x[144], 'x_145':landmark_x[145], 'x_146':landmark_x[146], 'x_147':landmark_x[147], 'x_148':landmark_x[148], 'x_149':landmark_x[149], 'x_150':landmark_x[150], 'x_151':landmark_x[151], 'x_152':landmark_x[152], 'x_153':landmark_x[153], 'x_154':landmark_x[154], 'x_155':landmark_x[155], 'x_156':landmark_x[156], 'x_157':landmark_x[157], 'x_158':landmark_x[158], 'x_159':landmark_x[159], 'x_160':landmark_x[160], 'x_161':landmark_x[161], 'x_162':landmark_x[162], 'x_163':landmark_x[163], 'x_164':landmark_x[164], 'x_165':landmark_x[165], 'x_166':landmark_x[166], 'x_167':landmark_x[167], 'x_168':landmark_x[168], 'x_169':landmark_x[169], 'x_170':landmark_x[170], 'x_171':landmark_x[171], 'x_172':landmark_x[172], 'x_173':landmark_x[173], 'x_174':landmark_x[174], 'x_175':landmark_x[175], 'x_176':landmark_x[176], 'x_177':landmark_x[177], 'x_178':landmark_x[178], 'x_179':landmark_x[179], 'x_180':landmark_x[180], 'x_181':landmark_x[181], 'x_182':landmark_x[182], 'x_183':landmark_x[183], 'x_184':landmark_x[184], 'x_185':landmark_x[185], 'x_186':landmark_x[186], 'x_187':landmark_x[187], 'x_188':landmark_x[188], 'x_189':landmark_x[189], 'x_190':landmark_x[190], 'x_191':landmark_x[191], 'x_192':landmark_x[192], 'x_193':landmark_x[193], 'x_194':landmark_x[194], 'x_195':landmark_x[195], 'x_196':landmark_x[196], 'x_197':landmark_x[197], 'x_198':landmark_x[198], 'x_199':landmark_x[199], 'x_200':landmark_x[200], 'x_201':landmark_x[201], 'x_202':landmark_x[202], 'x_203':landmark_x[203], 'x_204':landmark_x[204], 'x_205':landmark_x[205], 'x_206':landmark_x[206], 'x_207':landmark_x[207], 'x_208':landmark_x[208], 'x_209':landmark_x[209], 'x_210':landmark_x[210], 'x_211':landmark_x[211], 'x_212':landmark_x[212], 'x_213':landmark_x[213], 'x_214':landmark_x[214], 'x_215':landmark_x[215], 'x_216':landmark_x[216], 'x_217':landmark_x[217], 'x_218':landmark_x[218], 'x_219':landmark_x[219], 'x_220':landmark_x[220], 'x_221':landmark_x[221], 'x_222':landmark_x[222], 'x_223':landmark_x[223], 'x_224':landmark_x[224], 'x_225':landmark_x[225], 'x_226':landmark_x[226], 'x_227':landmark_x[227], 'x_228':landmark_x[228], 'x_229':landmark_x[229], 'x_230':landmark_x[230], 'x_231':landmark_x[231], 'x_232':landmark_x[232], 'x_233':landmark_x[233], 'x_234':landmark_x[234], 'x_235':landmark_x[235], 'x_236':landmark_x[236], 'x_237':landmark_x[237], 'x_238':landmark_x[238], 'x_239':landmark_x[239], 'x_240':landmark_x[240], 'x_241':landmark_x[241], 'x_242':landmark_x[242], 'x_243':landmark_x[243], 'x_244':landmark_x[244], 'x_245':landmark_x[245], 'x_246':landmark_x[246], 'x_247':landmark_x[247], 'x_248':landmark_x[248], 'x_249':landmark_x[249], 'x_250':landmark_x[250], 'x_251':landmark_x[251], 'x_252':landmark_x[252], 'x_253':landmark_x[253], 'x_254':landmark_x[254], 'x_255':landmark_x[255], 'x_256':landmark_x[256], 'x_257':landmark_x[257], 'x_258':landmark_x[258], 'x_259':landmark_x[259], 'x_260':landmark_x[260], 'x_261':landmark_x[261], 'x_262':landmark_x[262], 'x_263':landmark_x[263], 'x_264':landmark_x[264], 'x_265':landmark_x[265], 'x_266':landmark_x[266], 'x_267':landmark_x[267], 'x_268':landmark_x[268], 'x_269':landmark_x[269], 'x_270':landmark_x[270], 'x_271':landmark_x[271], 'x_272':landmark_x[272], 'x_273':landmark_x[273], 'x_274':landmark_x[274], 'x_275':landmark_x[275], 'x_276':landmark_x[276], 'x_277':landmark_x[277], 'x_278':landmark_x[278], 'x_279':landmark_x[279], 'x_280':landmark_x[280], 'x_281':landmark_x[281], 'x_282':landmark_x[282], 'x_283':landmark_x[283], 'x_284':landmark_x[284], 'x_285':landmark_x[285], 'x_286':landmark_x[286], 'x_287':landmark_x[287], 'x_288':landmark_x[288], 'x_289':landmark_x[289], 'x_290':landmark_x[290], 'x_291':landmark_x[291], 'x_292':landmark_x[292], 'x_293':landmark_x[293], 'x_294':landmark_x[294], 'x_295':landmark_x[295], 'x_296':landmark_x[296], 'x_297':landmark_x[297], 'x_298':landmark_x[298], 'x_299':landmark_x[299], 'x_300':landmark_x[300], 'x_301':landmark_x[301], 'x_302':landmark_x[302], 'x_303':landmark_x[303], 'x_304':landmark_x[304], 'x_305':landmark_x[305], 'x_306':landmark_x[306], 'x_307':landmark_x[307], 'x_308':landmark_x[308], 'x_309':landmark_x[309], 'x_310':landmark_x[310], 'x_311':landmark_x[311], 'x_312':landmark_x[312], 'x_313':landmark_x[313], 'x_314':landmark_x[314], 'x_315':landmark_x[315], 'x_316':landmark_x[316], 'x_317':landmark_x[317], 'x_318':landmark_x[318], 'x_319':landmark_x[319], 'x_320':landmark_x[320], 'x_321':landmark_x[321], 'x_322':landmark_x[322], 'x_323':landmark_x[323], 'x_324':landmark_x[324], 'x_325':landmark_x[325], 'x_326':landmark_x[326], 'x_327':landmark_x[327], 'x_328':landmark_x[328], 'x_329':landmark_x[329], 'x_330':landmark_x[330], 'x_331':landmark_x[331], 'x_332':landmark_x[332], 'x_333':landmark_x[333], 'x_334':landmark_x[334], 'x_335':landmark_x[335], 'x_336':landmark_x[336], 'x_337':landmark_x[337], 'x_338':landmark_x[338], 'x_339':landmark_x[339], 'x_340':landmark_x[340], 'x_341':landmark_x[341], 'x_342':landmark_x[342], 'x_343':landmark_x[343], 'x_344':landmark_x[344], 'x_345':landmark_x[345], 'x_346':landmark_x[346], 'x_347':landmark_x[347], 'x_348':landmark_x[348], 'x_349':landmark_x[349], 'x_350':landmark_x[350], 'x_351':landmark_x[351], 'x_352':landmark_x[352], 'x_353':landmark_x[353], 'x_354':landmark_x[354], 'x_355':landmark_x[355], 'x_356':landmark_x[356], 'x_357':landmark_x[357], 'x_358':landmark_x[358], 'x_359':landmark_x[359], 'x_360':landmark_x[360], 'x_361':landmark_x[361], 'x_362':landmark_x[362], 'x_363':landmark_x[363], 'x_364':landmark_x[364], 'x_365':landmark_x[365], 'x_366':landmark_x[366], 'x_367':landmark_x[367], 'x_368':landmark_x[368], 'x_369':landmark_x[369], 'x_370':landmark_x[370], 'x_371':landmark_x[371], 'x_372':landmark_x[372], 'x_373':landmark_x[373], 'x_374':landmark_x[374], 'x_375':landmark_x[375], 'x_376':landmark_x[376], 'x_377':landmark_x[377], 'x_378':landmark_x[378], 'x_379':landmark_x[379], 'x_380':landmark_x[380], 'x_381':landmark_x[381], 'x_382':landmark_x[382], 'x_383':landmark_x[383], 'x_384':landmark_x[384], 'x_385':landmark_x[385], 'x_386':landmark_x[386], 'x_387':landmark_x[387], 'x_388':landmark_x[388], 'x_389':landmark_x[389], 'x_390':landmark_x[390], 'x_391':landmark_x[391], 'x_392':landmark_x[392], 'x_393':landmark_x[393], 'x_394':landmark_x[394], 'x_395':landmark_x[395], 'x_396':landmark_x[396], 'x_397':landmark_x[397], 'x_398':landmark_x[398], 'x_399':landmark_x[399], 'x_400':landmark_x[400], 'x_401':landmark_x[401], 'x_402':landmark_x[402], 'x_403':landmark_x[403], 'x_404':landmark_x[404], 'x_405':landmark_x[405], 'x_406':landmark_x[406], 'x_407':landmark_x[407], 'x_408':landmark_x[408], 'x_409':landmark_x[409], 'x_410':landmark_x[410], 'x_411':landmark_x[411], 'x_412':landmark_x[412], 'x_413':landmark_x[413], 'x_414':landmark_x[414], 'x_415':landmark_x[415], 'x_416':landmark_x[416], 'x_417':landmark_x[417], 'x_418':landmark_x[418], 'x_419':landmark_x[419], 'x_420':landmark_x[420], 'x_421':landmark_x[421], 'x_422':landmark_x[422], 'x_423':landmark_x[423], 'x_424':landmark_x[424], 'x_425':landmark_x[425], 'x_426':landmark_x[426], 'x_427':landmark_x[427], 'x_428':landmark_x[428], 'x_429':landmark_x[429], 'x_430':landmark_x[430], 'x_431':landmark_x[431], 'x_432':landmark_x[432], 'x_433':landmark_x[433], 'x_434':landmark_x[434], 'x_435':landmark_x[435], 'x_436':landmark_x[436], 'x_437':landmark_x[437], 'x_438':landmark_x[438], 'x_439':landmark_x[439], 'x_440':landmark_x[440], 'x_441':landmark_x[441], 'x_442':landmark_x[442], 'x_443':landmark_x[443], 'x_444':landmark_x[444], 'x_445':landmark_x[445], 'x_446':landmark_x[446], 'x_447':landmark_x[447], 'x_448':landmark_x[448], 'x_449':landmark_x[449], 'x_450':landmark_x[450], 'x_451':landmark_x[451], 'x_452':landmark_x[452], 'x_453':landmark_x[453], 'x_454':landmark_x[454], 'x_455':landmark_x[455], 'x_456':landmark_x[456], 'x_457':landmark_x[457], 'x_458':landmark_x[458], 'x_459':landmark_x[459], 'x_460':landmark_x[460], 'x_461':landmark_x[461], 'x_462':landmark_x[462], 'x_463':landmark_x[463], 'x_464':landmark_x[464], 'x_465':landmark_x[465], 'x_466':landmark_x[466], 'x_467':landmark_x[467], 'y_0':landmark_y[0], 'y_1':landmark_y[1], 'y_2':landmark_y[2], 'y_3':landmark_y[3], 'y_4':landmark_y[4], 'y_5':landmark_y[5], 'y_6':landmark_y[6], 'y_7':landmark_y[7], 'y_8':landmark_y[8], 'y_9':landmark_y[9], 'y_10':landmark_y[10], 'y_11':landmark_y[11], 'y_12':landmark_y[12], 'y_13':landmark_y[13], 'y_14':landmark_y[14], 'y_15':landmark_y[15], 'y_16':landmark_y[16], 'y_17':landmark_y[17], 'y_18':landmark_y[18], 'y_19':landmark_y[19], 'y_20':landmark_y[20], 'y_21':landmark_y[21], 'y_22':landmark_y[22], 'y_23':landmark_y[23], 'y_24':landmark_y[24], 'y_25':landmark_y[25], 'y_26':landmark_y[26], 'y_27':landmark_y[27], 'y_28':landmark_y[28], 'y_29':landmark_y[29], 'y_30':landmark_y[30], 'y_31':landmark_y[31], 'y_32':landmark_y[32], 'y_33':landmark_y[33], 'y_34':landmark_y[34], 'y_35':landmark_y[35], 'y_36':landmark_y[36], 'y_37':landmark_y[37], 'y_38':landmark_y[38], 'y_39':landmark_y[39], 'y_40':landmark_y[40], 'y_41':landmark_y[41], 'y_42':landmark_y[42], 'y_43':landmark_y[43], 'y_44':landmark_y[44], 'y_45':landmark_y[45], 'y_46':landmark_y[46], 'y_47':landmark_y[47], 'y_48':landmark_y[48], 'y_49':landmark_y[49], 'y_50':landmark_y[50], 'y_51':landmark_y[51], 'y_52':landmark_y[52], 'y_53':landmark_y[53], 'y_54':landmark_y[54], 'y_55':landmark_y[55], 'y_56':landmark_y[56], 'y_57':landmark_y[57], 'y_58':landmark_y[58], 'y_59':landmark_y[59], 'y_60':landmark_y[60], 'y_61':landmark_y[61], 'y_62':landmark_y[62], 'y_63':landmark_y[63], 'y_64':landmark_y[64], 'y_65':landmark_y[65], 'y_66':landmark_y[66], 'y_67':landmark_y[67], 'y_68':landmark_y[68], 'y_69':landmark_y[69], 'y_70':landmark_y[70], 'y_71':landmark_y[71], 'y_72':landmark_y[72], 'y_73':landmark_y[73], 'y_74':landmark_y[74], 'y_75':landmark_y[75], 'y_76':landmark_y[76], 'y_77':landmark_y[77], 'y_78':landmark_y[78], 'y_79':landmark_y[79], 'y_80':landmark_y[80], 'y_81':landmark_y[81], 'y_82':landmark_y[82], 'y_83':landmark_y[83], 'y_84':landmark_y[84], 'y_85':landmark_y[85], 'y_86':landmark_y[86], 'y_87':landmark_y[87], 'y_88':landmark_y[88], 'y_89':landmark_y[89], 'y_90':landmark_y[90], 'y_91':landmark_y[91], 'y_92':landmark_y[92], 'y_93':landmark_y[93], 'y_94':landmark_y[94], 'y_95':landmark_y[95], 'y_96':landmark_y[96], 'y_97':landmark_y[97], 'y_98':landmark_y[98], 'y_99':landmark_y[99], 'y_100':landmark_y[100], 'y_101':landmark_y[101], 'y_102':landmark_y[102], 'y_103':landmark_y[103], 'y_104':landmark_y[104], 'y_105':landmark_y[105], 'y_106':landmark_y[106], 'y_107':landmark_y[107], 'y_108':landmark_y[108], 'y_109':landmark_y[109], 'y_110':landmark_y[110], 'y_111':landmark_y[111], 'y_112':landmark_y[112], 'y_113':landmark_y[113], 'y_114':landmark_y[114], 'y_115':landmark_y[115], 'y_116':landmark_y[116], 'y_117':landmark_y[117], 'y_118':landmark_y[118], 'y_119':landmark_y[119], 'y_120':landmark_y[120], 'y_121':landmark_y[121], 'y_122':landmark_y[122], 'y_123':landmark_y[123], 'y_124':landmark_y[124], 'y_125':landmark_y[125], 'y_126':landmark_y[126], 'y_127':landmark_y[127], 'y_128':landmark_y[128], 'y_129':landmark_y[129], 'y_130':landmark_y[130], 'y_131':landmark_y[131], 'y_132':landmark_y[132], 'y_133':landmark_y[133], 'y_134':landmark_y[134], 'y_135':landmark_y[135], 'y_136':landmark_y[136], 'y_137':landmark_y[137], 'y_138':landmark_y[138], 'y_139':landmark_y[139], 'y_140':landmark_y[140], 'y_141':landmark_y[141], 'y_142':landmark_y[142], 'y_143':landmark_y[143], 'y_144':landmark_y[144], 'y_145':landmark_y[145], 'y_146':landmark_y[146], 'y_147':landmark_y[147], 'y_148':landmark_y[148], 'y_149':landmark_y[149], 'y_150':landmark_y[150], 'y_151':landmark_y[151], 'y_152':landmark_y[152], 'y_153':landmark_y[153], 'y_154':landmark_y[154], 'y_155':landmark_y[155], 'y_156':landmark_y[156], 'y_157':landmark_y[157], 'y_158':landmark_y[158], 'y_159':landmark_y[159], 'y_160':landmark_y[160], 'y_161':landmark_y[161], 'y_162':landmark_y[162], 'y_163':landmark_y[163], 'y_164':landmark_y[164], 'y_165':landmark_y[165], 'y_166':landmark_y[166], 'y_167':landmark_y[167], 'y_168':landmark_y[168], 'y_169':landmark_y[169], 'y_170':landmark_y[170], 'y_171':landmark_y[171], 'y_172':landmark_y[172], 'y_173':landmark_y[173], 'y_174':landmark_y[174], 'y_175':landmark_y[175], 'y_176':landmark_y[176], 'y_177':landmark_y[177], 'y_178':landmark_y[178], 'y_179':landmark_y[179], 'y_180':landmark_y[180], 'y_181':landmark_y[181], 'y_182':landmark_y[182], 'y_183':landmark_y[183], 'y_184':landmark_y[184], 'y_185':landmark_y[185], 'y_186':landmark_y[186], 'y_187':landmark_y[187], 'y_188':landmark_y[188], 'y_189':landmark_y[189], 'y_190':landmark_y[190], 'y_191':landmark_y[191], 'y_192':landmark_y[192], 'y_193':landmark_y[193], 'y_194':landmark_y[194], 'y_195':landmark_y[195], 'y_196':landmark_y[196], 'y_197':landmark_y[197], 'y_198':landmark_y[198], 'y_199':landmark_y[199], 'y_200':landmark_y[200], 'y_201':landmark_y[201], 'y_202':landmark_y[202], 'y_203':landmark_y[203], 'y_204':landmark_y[204], 'y_205':landmark_y[205], 'y_206':landmark_y[206], 'y_207':landmark_y[207], 'y_208':landmark_y[208], 'y_209':landmark_y[209], 'y_210':landmark_y[210], 'y_211':landmark_y[211], 'y_212':landmark_y[212], 'y_213':landmark_y[213], 'y_214':landmark_y[214], 'y_215':landmark_y[215], 'y_216':landmark_y[216], 'y_217':landmark_y[217], 'y_218':landmark_y[218], 'y_219':landmark_y[219], 'y_220':landmark_y[220], 'y_221':landmark_y[221], 'y_222':landmark_y[222], 'y_223':landmark_y[223], 'y_224':landmark_y[224], 'y_225':landmark_y[225], 'y_226':landmark_y[226], 'y_227':landmark_y[227], 'y_228':landmark_y[228], 'y_229':landmark_y[229], 'y_230':landmark_y[230], 'y_231':landmark_y[231], 'y_232':landmark_y[232], 'y_233':landmark_y[233], 'y_234':landmark_y[234], 'y_235':landmark_y[235], 'y_236':landmark_y[236], 'y_237':landmark_y[237], 'y_238':landmark_y[238], 'y_239':landmark_y[239], 'y_240':landmark_y[240], 'y_241':landmark_y[241], 'y_242':landmark_y[242], 'y_243':landmark_y[243], 'y_244':landmark_y[244], 'y_245':landmark_y[245], 'y_246':landmark_y[246], 'y_247':landmark_y[247], 'y_248':landmark_y[248], 'y_249':landmark_y[249], 'y_250':landmark_y[250], 'y_251':landmark_y[251], 'y_252':landmark_y[252], 'y_253':landmark_y[253], 'y_254':landmark_y[254], 'y_255':landmark_y[255], 'y_256':landmark_y[256], 'y_257':landmark_y[257], 'y_258':landmark_y[258], 'y_259':landmark_y[259], 'y_260':landmark_y[260], 'y_261':landmark_y[261], 'y_262':landmark_y[262], 'y_263':landmark_y[263], 'y_264':landmark_y[264], 'y_265':landmark_y[265], 'y_266':landmark_y[266], 'y_267':landmark_y[267], 'y_268':landmark_y[268], 'y_269':landmark_y[269], 'y_270':landmark_y[270], 'y_271':landmark_y[271], 'y_272':landmark_y[272], 'y_273':landmark_y[273], 'y_274':landmark_y[274], 'y_275':landmark_y[275], 'y_276':landmark_y[276], 'y_277':landmark_y[277], 'y_278':landmark_y[278], 'y_279':landmark_y[279], 'y_280':landmark_y[280], 'y_281':landmark_y[281], 'y_282':landmark_y[282], 'y_283':landmark_y[283], 'y_284':landmark_y[284], 'y_285':landmark_y[285], 'y_286':landmark_y[286], 'y_287':landmark_y[287], 'y_288':landmark_y[288], 'y_289':landmark_y[289], 'y_290':landmark_y[290], 'y_291':landmark_y[291], 'y_292':landmark_y[292], 'y_293':landmark_y[293], 'y_294':landmark_y[294], 'y_295':landmark_y[295], 'y_296':landmark_y[296], 'y_297':landmark_y[297], 'y_298':landmark_y[298], 'y_299':landmark_y[299], 'y_300':landmark_y[300], 'y_301':landmark_y[301], 'y_302':landmark_y[302], 'y_303':landmark_y[303], 'y_304':landmark_y[304], 'y_305':landmark_y[305], 'y_306':landmark_y[306], 'y_307':landmark_y[307], 'y_308':landmark_y[308], 'y_309':landmark_y[309], 'y_310':landmark_y[310], 'y_311':landmark_y[311], 'y_312':landmark_y[312], 'y_313':landmark_y[313], 'y_314':landmark_y[314], 'y_315':landmark_y[315], 'y_316':landmark_y[316], 'y_317':landmark_y[317], 'y_318':landmark_y[318], 'y_319':landmark_y[319], 'y_320':landmark_y[320], 'y_321':landmark_y[321], 'y_322':landmark_y[322], 'y_323':landmark_y[323], 'y_324':landmark_y[324], 'y_325':landmark_y[325], 'y_326':landmark_y[326], 'y_327':landmark_y[327], 'y_328':landmark_y[328], 'y_329':landmark_y[329], 'y_330':landmark_y[330], 'y_331':landmark_y[331], 'y_332':landmark_y[332], 'y_333':landmark_y[333], 'y_334':landmark_y[334], 'y_335':landmark_y[335], 'y_336':landmark_y[336], 'y_337':landmark_y[337], 'y_338':landmark_y[338], 'y_339':landmark_y[339], 'y_340':landmark_y[340], 'y_341':landmark_y[341], 'y_342':landmark_y[342], 'y_343':landmark_y[343], 'y_344':landmark_y[344], 'y_345':landmark_y[345], 'y_346':landmark_y[346], 'y_347':landmark_y[347], 'y_348':landmark_y[348], 'y_349':landmark_y[349], 'y_350':landmark_y[350], 'y_351':landmark_y[351], 'y_352':landmark_y[352], 'y_353':landmark_y[353], 'y_354':landmark_y[354], 'y_355':landmark_y[355], 'y_356':landmark_y[356], 'y_357':landmark_y[357], 'y_358':landmark_y[358], 'y_359':landmark_y[359], 'y_360':landmark_y[360], 'y_361':landmark_y[361], 'y_362':landmark_y[362], 'y_363':landmark_y[363], 'y_364':landmark_y[364], 'y_365':landmark_y[365], 'y_366':landmark_y[366], 'y_367':landmark_y[367], 'y_368':landmark_y[368], 'y_369':landmark_y[369], 'y_370':landmark_y[370], 'y_371':landmark_y[371], 'y_372':landmark_y[372], 'y_373':landmark_y[373], 'y_374':landmark_y[374], 'y_375':landmark_y[375], 'y_376':landmark_y[376], 'y_377':landmark_y[377], 'y_378':landmark_y[378], 'y_379':landmark_y[379], 'y_380':landmark_y[380], 'y_381':landmark_y[381], 'y_382':landmark_y[382], 'y_383':landmark_y[383], 'y_384':landmark_y[384], 'y_385':landmark_y[385], 'y_386':landmark_y[386], 'y_387':landmark_y[387], 'y_388':landmark_y[388], 'y_389':landmark_y[389], 'y_390':landmark_y[390], 'y_391':landmark_y[391], 'y_392':landmark_y[392], 'y_393':landmark_y[393], 'y_394':landmark_y[394], 'y_395':landmark_y[395], 'y_396':landmark_y[396], 'y_397':landmark_y[397], 'y_398':landmark_y[398], 'y_399':landmark_y[399], 'y_400':landmark_y[400], 'y_401':landmark_y[401], 'y_402':landmark_y[402], 'y_403':landmark_y[403], 'y_404':landmark_y[404], 'y_405':landmark_y[405], 'y_406':landmark_y[406], 'y_407':landmark_y[407], 'y_408':landmark_y[408], 'y_409':landmark_y[409], 'y_410':landmark_y[410], 'y_411':landmark_y[411], 'y_412':landmark_y[412], 'y_413':landmark_y[413], 'y_414':landmark_y[414], 'y_415':landmark_y[415], 'y_416':landmark_y[416], 'y_417':landmark_y[417], 'y_418':landmark_y[418], 'y_419':landmark_y[419], 'y_420':landmark_y[420], 'y_421':landmark_y[421], 'y_422':landmark_y[422], 'y_423':landmark_y[423], 'y_424':landmark_y[424], 'y_425':landmark_y[425], 'y_426':landmark_y[426], 'y_427':landmark_y[427], 'y_428':landmark_y[428], 'y_429':landmark_y[429], 'y_430':landmark_y[430], 'y_431':landmark_y[431], 'y_432':landmark_y[432], 'y_433':landmark_y[433], 'y_434':landmark_y[434], 'y_435':landmark_y[435], 'y_436':landmark_y[436], 'y_437':landmark_y[437], 'y_438':landmark_y[438], 'y_439':landmark_y[439], 'y_440':landmark_y[440], 'y_441':landmark_y[441], 'y_442':landmark_y[442], 'y_443':landmark_y[443], 'y_444':landmark_y[444], 'y_445':landmark_y[445], 'y_446':landmark_y[446], 'y_447':landmark_y[447], 'y_448':landmark_y[448], 'y_449':landmark_y[449], 'y_450':landmark_y[450], 'y_451':landmark_y[451], 'y_452':landmark_y[452], 'y_453':landmark_y[453], 'y_454':landmark_y[454], 'y_455':landmark_y[455], 'y_456':landmark_y[456], 'y_457':landmark_y[457], 'y_458':landmark_y[458], 'y_459':landmark_y[459], 'y_460':landmark_y[460], 'y_461':landmark_y[461], 'y_462':landmark_y[462], 'y_463':landmark_y[463], 'y_464':landmark_y[464], 'y_465':landmark_y[465], 'y_466':landmark_y[466], 'y_467':landmark_y[467]}, index=[0])
                    video_df = pd.concat([video_df, landmarks_df], ignore_index=True)

        cap.release()
        video_df.to_csv(os.path.join(out_dir, f'{videoName}.csv'), index=False)
