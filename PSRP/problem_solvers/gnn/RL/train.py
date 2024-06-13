import time

from tqdm import tqdm

from PSRP.problem_solvers.gnn.RL.envipoment import IRPEnv_Custom

def train(
        agent,
        train_dataset,
        parameters_dict,
        check_point_dir: str = "./check_points/",
    ):
        epochs = parameters_dict['epochs']
        k_vehicles = parameters_dict['k_vehicles']
        max_trips = parameters_dict['max_trips']
        eval_epochs = parameters_dict['eval_epochs']
        num_nodes = parameters_dict['num_nodes']
        products_count = parameters_dict['products_count']

        start_time = time.time()

        loss_history = []
        dist_history = []
        dry_runs_history = []

        for e in tqdm(range(epochs), total=epochs):
          for batch in train_dataset:
            env = IRPEnv_Custom(batch, parameters_dict)

            agent.model.train()

            loss_m, loss_b, log_prob, kpi = agent.step(env, (False, True))
            advantage = (loss_m - loss_b) * -1
            loss = (advantage * log_prob).mean()

            loss_history.append(loss.mean().item())

            dry_runs_loss = 10 * env.loss_dry_runs 

            dry_runs_history.append(dry_runs_loss.mean().item())
            dist_history.append(-1*loss_m.mean().item())

            # backpropagate
            agent.opt.zero_grad()
            loss.backward()

            agent.opt.step()

            # update model if better
            env = IRPEnv_Custom(batch, parameters_dict)

            baseline_replaced = agent.baseline_update(env, eval_epochs)
            agent.save_model(episode=e, check_point_dir=check_point_dir)
        return loss_history, dist_history, dry_runs_history, kpi