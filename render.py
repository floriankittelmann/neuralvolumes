from utils.ImportConfigUtil import ImportConfigUtil

from utils.RenderUtils import RenderUtils

if __name__ == "__main__":
    render_utils = RenderUtils()
    args = render_utils.parse_cmd_arguments()
    outpath = render_utils.get_outpath_and_print_infos(args)

    import_config_util = ImportConfigUtil()
    experconfig = import_config_util.import_module(args.experconfig)
    render_profile = experconfig.DatasetConfig().get_render_profile()

    # eval
    if args.cam == "all":
        for camera_nr in range(36):
            camera = "{:03d}".format(camera_nr)
            print("start with camera " + camera)
            render_profile.cam = camera
            render_utils.render(render_profile, args, outpath)
    else:
        render_profile.cam = args.cam
        render_utils.render(render_profile, args, outpath)

