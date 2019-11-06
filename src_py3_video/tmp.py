config = {}
config["num_iterations"] = 31000
config["test_interval"] = 500
config["prep"] = [{"name":"source", "type":"image", "test_10crop":True, "resize_size":256, "crop_size":224}, {"name":"target", "type":"image", "test_10crop":True, "resize_size":256, "crop_size":224}]
config["loss"] = {"name":args.loss_name, "trade_off":args.tradeoff }
config["data"] = [{"name":"source", "type":"image", "list_path":{"train":"../data/office/POM/"+args.source+"_videodata_list.txt"},"batch_size":{"train":36} },
                {"name":"target", "type":"image", "list_path":{"train":"../data/office/VR/"+args.target+"_videodata_list.txt", "validation":"../data/office/VR/"+args.target_val+"_videodata_list.txt"},
                "batch_size":{"train":36,"validation":4} }]
config["network"] = {"name":"ResNet18", "use_bottleneck":args.using_bottleneck, "bottleneck_dim":256}
config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.5, "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.03, "gamma":0.0003, "power":0.75} }
print (config["loss"])
print(config["data"][0]["list_path"]["train"])
print("Learning rate:: 0.01 with 1 bottleneck layers")
