# exec(open('learn2.py').read())
# exec(open('small_leanr1.py').read())
_, img = vid.read()
if img is None:
	logging.warning("Empty Frame")
	time.sleep(0.1)
	count+=1
	if count < 3:
		print("continue")
	else: 
		print("break")
print("Pos1")
img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img_in = tf.expand_dims(img_in, 0)
img_in = transform_images(img_in, 416)
t1 = time.time()
boxes, scores, classes, nums = yolo.predict(img_in)
classes = classes[0]
names = []
print("Pos2")
for i in range(len(classes)):
	names.append(class_names[int(classes[i])])
names = np.array(names)
converted_boxes = convert_boxes(img, boxes[0])
features = encoder(img, converted_boxes)    
detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]    
#initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
# run non-maxima suppresion
boxs = np.array([d.tlwh for d in detections])
scores = np.array([d.confidence for d in detections])
classes = np.array([d.class_name for d in detections])
indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
detections = [detections[i] for i in indices]        
# Call the tracker
tracker.predict()
tracker.update(detections)
print("Pos3")
for track in tracker.tracks:
	if not track.is_confirmed() or track.time_since_update > 1:
		print("continue") 
	bbox = track.to_tlbr()
	class_name = track.get_class()
	color = colors[int(track.track_id) % len(colors)]
	color = [i * 255 for i in color]
	if track.track_id == 1:
		print("Features: "+str(len(track.features)))
		print("center at " + str((int(bbox[0]) + int(bbox[2]))/2.0) + ", " + str((int(bbox[1]) + int(bbox[3]))/2.0))
		cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
		cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
fps  = ( fps + (1./(time.time()-t1)) ) / 2
cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
#cv2.rectangle(img, (200, 500), (400, 100), color, 2)
line_thickness = 20
cv2.line(img, (140, 400), (1700, 825), color,thickness=line_thickness)
cv2.imshow('output', img)
print("Pos4")
out.write(img)
frame_index = frame_index + 1
list_file.write(str(frame_index)+' ')
if len(converted_boxes) != 0:
	for i in range(0,len(converted_boxes)):
		list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
list_file.write('\n')
if cv2.waitKey(1) == ord('q'):
	print("break")