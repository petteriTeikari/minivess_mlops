import os
from loguru import logger
from pathlib import Path


def debug_mounts(mounts_list: list, try_to_write: bool = True):
    logger.info(
        "Username = {}, UID = {}, GID = {}".format(
            os.getenv("USER"), os.getuid(), os.getgid()
        )
    )
    # mounts_list = ['/home/petteri/artifacts']
    for mount in mounts_list:
        logger.debug("MOUNT: {}".format(mount))
        if not os.path.exists(mount):
            logger.error('Mount "{}" does not exist!'.format(mount))
            raise IOError('Mount "{}" does not exist!'.format(mount))
        path = Path(mount)
        owner = path.owner()
        group = path.group()
        logger.debug(f" owned by {owner}:{group} (owner:group)")
        mount_obj = os.stat(mount)
        oct_perm = oct(mount_obj.st_mode)[-4:]
        logger.debug(f" owned by {mount_obj.st_uid}:{mount_obj.st_gid} (owner:group)")
        logger.debug(f" mount permissions: {oct_perm}")
        logger.debug(
            " read access = {}".format(os.access(mount, os.R_OK))
        )  # Check for read access
        logger.debug(
            " write access = {}".format(os.access(mount, os.W_OK))
        )  # Check for write access
        logger.debug(
            " execution access = {}".format(os.access(mount, os.X_OK))
        )  # Check for execution access
        logger.debug(
            " existence of dir = {}".format(os.access(mount, os.F_OK))
        )  # Check for existence of file

        if os.access(mount, os.W_OK):
            logger.debug("Trying to write to the mount (write access was OK)")
            path_out = os.path.join(mount, "test_write.txt")

            if os.path.exists(path_out):
                # unlike normal filesystem, mountpoint-s3 does not allow overwriting files,
                # so if it exists already we need to delete it first
                logger.info(
                    "File {} already exists, deleting it first".format(path_out)
                )
                try:
                    os.remove(path_out)
                except Exception as e:
                    logger.error("Problem deleting file {}, e = {}".format(path_out, e))
                    raise IOError(
                        "Problem deleting file {}, e = {}".format(path_out, e)
                    )

            try:
                file1 = open(path_out, "w")
                file1.write("Hello debug world!")
                file1.close()
                logger.debug("File write succesful!")
                mount_obj = os.stat(path_out)
                logger.debug(
                    " file_permission = {}, {}:{}".format(
                        oct(os.stat(path_out).st_mode)[-4:],
                        mount_obj.st_uid,
                        mount_obj.st_gid,
                    )
                )
            except Exception as e:
                logger.error(
                    "Problem with file write to {}, e = {}".format(path_out, e)
                )
                raise IOError(
                    "Problem with file write to {}, e = {}".format(path_out, e)
                )

            if os.path.exists(path_out):
                try:
                    os.remove(path_out)
                    logger.debug("File delete succesful!")
                except Exception as e:
                    logger.error("Problem deleting file {}, e = {}".format(path_out, e))
                    raise IOError(
                        "Problem deleting file {}, e = {}".format(path_out, e)
                    )
            else:
                logger.debug(
                    "Weirdly you do not have any file to delete even though write went through OK?"
                )
