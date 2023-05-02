
from trainer.fsl_trainer import FSLTrainer

from trainer.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)
