import torch.cuda
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from configure import *
from hierarchical import *


import time

CUDA_LAUNCH_BLOCKING = 1.

if __name__ == "__main__":
    args = parse_args()
    args, architecture, preds, train_dataset, test_dataset = configure(args)

    # torch.cuda.set_device(args.dev_num)
    # print(torch.cuda.current_device())

    stem = args.stem
    data_path = args.data_path
    seeds = args.seeds   # 52/13/22/2
    loss_criterion = args.loss_criterion  # {'CwC', 'CwC_CE', 'PvN', 'CWG'}"
    dataset = args.dataset
    ILT = args.ILT  # {Acc, Fast}
    save_ = args.save
    CFSE = args.CFSE  # {False = FFCNN}
    sf_pred = args.sf_pred
    ClassGroup = args.ClassGroup
    retrain = args.retrain
    stage = args.stage
    batch_size = args.batch_size
    show_iters = args.show_iters
    n_epochs_d = args.n_epochs
    N_Classes = args.N_Classes
    min_testerror = args.min_testerror
    log_root = args.log_root
    log_steps = args.log_steps
    online_logger = args.online_logger
    run_name = args.run_name
    # flow = args.conf_flow
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    flows = ['RCB', 'CBR', 'CRB']
    powers = [2]   # 2.1
    betas = [[0, 0]] #, [1, 1], [1, 0], [0.5, 0.5], [0, 1]]

    num_SG_Layers = [2]
    stems = ['WAN']  #'ResNet']
    # for tk in range(len(powers)):
    print(dataset)
    for seed in seeds:
        seed_everything(seed=seed)
        for beta_set in betas:
            for SG_Layers in num_SG_Layers:
                flow = flows[1]
                power = 2  #powers[tk]
                print(flow)
                now = datetime.now()
                date_time = now.strftime("%Y%m%d_%H%M%S")

                if stem == 'WAN':
                    if dataset == 'imgnet200':
                        channels_list = [100, 200, 400, 400, 800, 800, 1600]
                    else:
                        channels_list = [100, 200, 400, 400, 800, 1600]
                    model = Hier_CwC_WAN(channels_list, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion,
                                            N_Classes=N_Classes, flow=flow, n_epochs=n_epochs_d, skip_mode='add', num_supergroup_layers=SG_Layers, beta_start=beta_set[0], beta_end=beta_set[1])
                else:
                    if dataset == 'imgnet200':
                        channels_list = [100, 200, 200, 200, 200, 400, 400, 800, 800, 1600]
                    else:
                        channels_list = [100, 100, 100, 200, 200, 400, 400, 800, 800, 1600]
                    model = Hier_CwC_ResNet(channels_list, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion,
                                            N_Classes=N_Classes, flow=flow, n_epochs=n_epochs_d, skip_mode='add', num_supergroup_layers=SG_Layers, beta_start=beta_set[0], beta_end=beta_set[1])

                model_id = (
                    stem
                    + f'_{SG_Layers}SG_{len(channels_list)-SG_Layers}CwC'
                    + '_'
                    + dataset
                    + '_Ch'
                    + 'n'.join(map(str, channels_list))
                    + f'_seed{seed}_{date_time}'
                )
                resolved_run_name = run_name or model_id
                logger = ExperimentLogger(
                    log_root=log_root,
                    stem=stem,
                    dataset=dataset,
                    run_name=resolved_run_name,
                    seed=seed,
                    args_dict=vars(args),
                    model=model,
                    online_logger=online_logger,
                    command=" ".join(sys.argv),
                )
                #  Main Training Loop
                print(len(train_loader))
                print(model)
                print(model_id)
                print(f'Run directory: {logger.run_dir}')
                if retrain:
                    model = load_model(model, model_id, dataset, stage, param=True)

                #  Main Training Loop

                Avg_train_losses = []
                Avg_test_losses = []
                Sf_train_losses = []
                Sf_test_losses = []
                layerwise_loss = []
                n_epochs_d = model.start_end[len(model.conv_layers) - 1][1] + 1
                # after model creation

                for epoch in range(n_epochs_d):

                    sf_epoch_losses = []
                    ep_layer_l = []
                    epoch_tr_errors = []

                    print('\n')
                    print('-- Epoch: {} ------------------------------------'.format(epoch))

                    model.train()

                    b4_train = time.time()
                    train_t = 0
                    test_t = 0
                    load_t = 0
                    t3 = 0
                    epoch_start = time.time()

                    start_end = model.start_end
                    # Build masks once per epoch
                    train_mask = []
                    started_mask = []
                    for i, (s, e) in enumerate(model.start_end):
                        started = (epoch >= s)
                        train_this = (s <= epoch < e)
                        started_mask.append(started)
                        train_mask.append(train_this)

                    # Set modes + requires_grad
                    for i, layer in enumerate(model.conv_layers):
                        if train_mask[i]:
                            layer.train(True)
                            for p in layer.parameters():
                                p.requires_grad_(True)
                        else:
                            layer.train(False)  # equivalent to layer.eval()
                            for p in layer.parameters():
                                p.requires_grad_(False)

                    se_time = time.time()
                    for step, (x, y) in enumerate(train_loader):

                        collect_pred = (step > len(train_loader) - len(test_loader))
                        layer_errs = []

                        t0 = time.time()

                        x = Variable(x).cuda()
                        y = y.long().cuda()

                        h = x

                        t1 = time.time()
                        if step != 0:
                            load_t += (t1 - t2)

                        skip = {}
                        h = x


                        for i, layer in enumerate(model.conv_layers):

                            s, e = model.start_end[i]
                            if epoch < s:
                                layer_errs.append(1.1)
                                continue

                            if i in model.skip_to:
                                h = model._apply_skip(h, skip[f"skip_{i - 2}"])

                            #     sk = skip[f"skip_{i - 2}"]
                            #     if sk.shape[2:] != h.shape[2:]:
                            #         sk = model.match_spatial(sk, h)
                            #         h = torch.cat((h, sk), dim=1)

                            if i in model.skip_from:
                                skip[f"skip_{i}"] = h.detach()  # store detached to avoid holding graphs

                            if s <= epoch < e:  # given that above it continues if epoch < s, then: s <= epoch < e
                                h, g = layer.learn(h, y, False)  # training
                                layer_step_error = layer.eval_pred(g, y, eval=False)
                            else: # means that epoch >= e
                                h, _ = layer.forward(h, eval=True, compute_gf=False)
                                layer_step_error = layer.last_tr_pred

                            layer_errs.append(layer_step_error)

                        model.iter += 1
                        # model.train_(x, y, epoch)
                        t2 = time.time()
                        train_t += (t2 - t1)
                        epoch_tr_errors.append(layer_errs)

                        if log_steps > 0 and (step % log_steps == 0):
                            step_metrics = {
                                'batch_train_error_mean': float(np.mean(layer_errs)),
                                'batch_train_error_last_layer': float(layer_errs[-1]),
                                'data_load_seconds': float(load_t),
                                'train_iteration_seconds': float(train_t),
                            }
                            for li, layer_err in enumerate(layer_errs):
                                step_metrics[f'layer_{li}_train_error'] = float(layer_err)
                            logger.log_step(epoch=epoch, step=step, split='train', metrics=step_metrics)


                    post_train = time.time()
                    epoch_te_errors = []

                    for i, conv_layer in enumerate(model.conv_layers):
                        if train_mask[i]:
                            conv_layer.scheduler.step()

                    model.eval()
                    t3 = time.time()
                    with torch.no_grad():
                        for step, (x_p, y_p) in enumerate(test_loader):
                            x_p = Variable(x_p).cuda()
                            y_p = y_p.long().cuda()

                            layer_pred = model.predict(x_p, y_p, epoch)
                            epoch_te_errors.append(layer_pred)


                    # # PRINTS FOR EVERY EPOCH
                    print('Epoch: {}'.format(epoch))

                    post_test = time.time()
                    test_t += (post_test - t3)

                    ep_train_error_list = []
                    ep_test_error_list = []
                    epoch_lrs = []
                    for i, layer in enumerate(model.conv_layers):

                        ep_train_error = np.asarray(epoch_tr_errors)[:, i].mean()
                        ep_test_error = np.asarray(epoch_te_errors)[:, i].mean()
                        ep_train_error_list.append(ep_train_error)
                        ep_test_error_list.append(ep_test_error)
                        epoch_lrs.append(layer.opt.param_groups[0]["lr"])

                        print('Layer: {} - Avg Pred - Train Error = {}'.format(i, ep_train_error))
                        print('          - Avg Pred - Test  Error = {}'.format(ep_test_error))
                        info = f'|| lr: {layer.opt.param_groups[0]["lr"]:.5f}'
                        print(info)

                        layer_loss = layer.epoch_loss()
                        ep_layer_l.append(layer_loss)
                        print('Conv Layer_{} Loss : {}'.format(i, layer_loss))

                    for i, layer in enumerate(model.nn_layers):
                        layer_loss = layer.epoch_loss()
                        ep_layer_l.append(layer_loss)
                        print('          NN Layer_{} Loss : {}'.format(i, layer_loss))

                    layerwise_loss.append(ep_layer_l)

                    Avg_train_losses.append(ep_train_error_list)
                    Avg_test_losses.append(ep_test_error_list)

                    # SAVE MODEL
                    if min_testerror > min(ep_test_error_list) and save_:
                        best_idx = int(np.argmin(ep_test_error_list))
                        save_state(model, model_id, dataset, stage + 1)
                        ckpt_path = logger.save_checkpoint(
                            model,
                            dataset,
                            f'best_stage{stage + 1}_epoch{epoch}',
                            score=float(min(ep_test_error_list)),
                            max_to_keep=2,
                        )
                        logger.update_summary(best_checkpoint=ckpt_path, best_epoch=epoch, best_layer=best_idx,
                                              best_test_error=float(min(ep_test_error_list)))
                        min_testerror = min(ep_test_error_list)

                    torch.cuda.empty_cache()

                    final_epoch_time = time.time()
                    # b4_train, se_time, post_train, post_test, final_epoch_time
                    # print('Preparing time: {}'.format(se_time - b4_train))
                    print('Load Time: {}'.format(load_t))
                    print('Train Iteration: {}'.format(train_t))
                    print('Predict Iteration: {}'.format(test_t))
                    print('Total Train Loop: {}'.format(post_train - se_time))
                    print('Total Test Loop: {}'.format(post_test - post_train))
                    print('Stats time: {}'.format(final_epoch_time - post_test))
                    print('Total epoch time: {}'.format(final_epoch_time - b4_train))

                    epoch_record = {
                        'epoch_seconds': float(final_epoch_time - epoch_start),
                        'load_seconds': float(load_t),
                        'train_seconds': float(train_t),
                        'test_seconds': float(test_t),
                        'train_error_mean': float(np.mean(ep_train_error_list)),
                        'test_error_mean': float(np.mean(ep_test_error_list)),
                        'best_test_error_epoch': float(np.min(ep_test_error_list)),
                        'best_test_layer_epoch': int(np.argmin(ep_test_error_list)),
                        'layer_train_errors': [float(x) for x in ep_train_error_list],
                        'layer_test_errors': [float(x) for x in ep_test_error_list],
                        'layer_losses': [float(x) for x in ep_layer_l],
                        'layer_lrs': [float(x) for x in epoch_lrs],
                    }
                    logger.log_epoch(epoch=epoch, metrics=epoch_record)

                pred_losses = np.concatenate((Avg_train_losses, Avg_test_losses), 1)  # , gd_train_losses, gd_test_losses]
                predictor_path = logger.save_array(pred_losses, 'predictor_losses', layer_losses=False)
                layerwise_path = logger.save_array(layerwise_loss, 'layerwise_losses', layer_losses=True)
                logger.update_summary(
                    final_min_test_error=float(np.min(Avg_test_losses)),
                    predictor_losses_csv=predictor_path,
                    layerwise_losses_csv=layerwise_path,
                    n_epochs_ran=int(n_epochs_d),
                    model_id=model_id,
                )
                logger.close()

