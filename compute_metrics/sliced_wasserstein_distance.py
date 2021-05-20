import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="semantic")
    args = parser.parse_args()
    
    model_name = args.model_name
    current_path = os.path.dirname(__file__)
    script_path = os.path.join(current_path,"swd", "inria_swd.py")

    swd_script = f"python {script_path} --o /SSSGAN/frechet_ds/images --f /SSSSGAN/results/{model_name}/test_latest/images/synthesized_image"
    print("Compute swd")
    os.system(swd_script)