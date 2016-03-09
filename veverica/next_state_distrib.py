def next_state_distrib(state, new_edges, priors):
	distrib = 16 * [0, ]
	nu, nv = new_edges
	pinp, pinn, poutp, poutn = priors
	kind = (min(nu, 2), min(nv, 2))
	if kind == (0, 0):
		distrib[state] = 1
		return distrib
	if kind == (0, 1):
		if state == 0:
			distrib[1] = poutn; distrib[2] = poutp
			return distrib
		if state == 1:
			distrib[1] = poutn; distrib[3] = poutp
			return distrib
		if state == 2:
			distrib[2] = poutp; distrib[3] = poutn
			return distrib
		if state == 3:
			distrib[3] = 1
			return distrib
		if state == 4:
			distrib[5] = poutn; distrib[6] = poutp
			return distrib
		if state == 5:
			distrib[5] = poutn; distrib[7] = poutp
			return distrib
		if state == 6:
			distrib[6] = poutp; distrib[7] = poutn
			return distrib
		if state == 7:
			distrib[7] = 1
			return distrib
		if state == 8:
			distrib[9] = poutn; distrib[10] = poutp
			return distrib
		if state == 9:
			distrib[9] = poutn; distrib[11] = poutp
			return distrib
		if state == 10:
			distrib[10] = poutp; distrib[11] = poutn
			return distrib
		if state == 11:
			distrib[11] = 1
			return distrib
		if state == 12:
			distrib[13] = poutn; distrib[14] = poutp
			return distrib
		if state == 13:
			distrib[13] = poutn; distrib[15] = poutp
			return distrib
		if state == 14:
			distrib[14] = poutp; distrib[15] = poutn
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (0, 2):
		if state == 0:
			distrib[1] = poutn**nv; distrib[2] = poutp**nv; distrib[3] = -poutn**nv - poutp**nv + 1
			return distrib
		if state == 1:
			distrib[1] = poutn**nv; distrib[3] = -poutn**nv + 1
			return distrib
		if state == 2:
			distrib[2] = poutp**nv; distrib[3] = -poutp**nv + 1
			return distrib
		if state == 3:
			distrib[3] = 1
			return distrib
		if state == 4:
			distrib[5] = poutn**nv; distrib[6] = poutp**nv; distrib[7] = -poutn**nv - poutp**nv + 1
			return distrib
		if state == 5:
			distrib[5] = poutn**nv; distrib[7] = -poutn**nv + 1
			return distrib
		if state == 6:
			distrib[6] = poutp**nv; distrib[7] = -poutp**nv + 1
			return distrib
		if state == 7:
			distrib[7] = 1
			return distrib
		if state == 8:
			distrib[9] = poutn**nv; distrib[10] = poutp**nv; distrib[11] = -poutn**nv - poutp**nv + 1
			return distrib
		if state == 9:
			distrib[9] = poutn**nv; distrib[11] = -poutn**nv + 1
			return distrib
		if state == 10:
			distrib[10] = poutp**nv; distrib[11] = -poutp**nv + 1
			return distrib
		if state == 11:
			distrib[11] = 1
			return distrib
		if state == 12:
			distrib[13] = poutn**nv; distrib[14] = poutp**nv; distrib[15] = -poutn**nv - poutp**nv + 1
			return distrib
		if state == 13:
			distrib[13] = poutn**nv; distrib[15] = -poutn**nv + 1
			return distrib
		if state == 14:
			distrib[14] = poutp**nv; distrib[15] = -poutp**nv + 1
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (1, 0):
		if state == 0:
			distrib[4] = pinn; distrib[8] = pinp
			return distrib
		if state == 1:
			distrib[5] = pinn; distrib[9] = pinp
			return distrib
		if state == 2:
			distrib[6] = pinn; distrib[10] = pinp
			return distrib
		if state == 3:
			distrib[7] = pinn; distrib[11] = pinp
			return distrib
		if state == 4:
			distrib[4] = pinn; distrib[12] = pinp
			return distrib
		if state == 5:
			distrib[5] = pinn; distrib[13] = pinp
			return distrib
		if state == 6:
			distrib[6] = pinn; distrib[14] = pinp
			return distrib
		if state == 7:
			distrib[7] = pinn; distrib[15] = pinp
			return distrib
		if state == 8:
			distrib[8] = pinp; distrib[12] = pinn
			return distrib
		if state == 9:
			distrib[9] = pinp; distrib[13] = pinn
			return distrib
		if state == 10:
			distrib[10] = pinp; distrib[14] = pinn
			return distrib
		if state == 11:
			distrib[11] = pinp; distrib[15] = pinn
			return distrib
		if state == 12:
			distrib[12] = 1
			return distrib
		if state == 13:
			distrib[13] = 1
			return distrib
		if state == 14:
			distrib[14] = 1
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (1, 1):
		if state == 0:
			distrib[5] = pinn*poutn; distrib[6] = pinn*poutp; distrib[9] = pinp*poutn
			distrib[10] = pinp*poutp
			return distrib
		if state == 1:
			distrib[5] = pinn*poutn; distrib[7] = pinn*poutp; distrib[9] = pinp*poutn
			distrib[11] = pinp*poutp
			return distrib
		if state == 2:
			distrib[6] = pinn*poutp; distrib[7] = pinn*poutn; distrib[10] = pinp*poutp
			distrib[11] = pinp*poutn
			return distrib
		if state == 3:
			distrib[7] = pinn; distrib[11] = pinp
			return distrib
		if state == 4:
			distrib[5] = pinn*poutn; distrib[6] = pinn*poutp; distrib[13] = pinp*poutn
			distrib[14] = pinp*poutp
			return distrib
		if state == 5:
			distrib[5] = pinn*poutn; distrib[7] = pinn*poutp; distrib[13] = pinp*poutn
			distrib[15] = pinp*poutp
			return distrib
		if state == 6:
			distrib[6] = pinn*poutp; distrib[7] = pinn*poutn; distrib[14] = pinp*poutp
			distrib[15] = pinp*poutn
			return distrib
		if state == 7:
			distrib[7] = pinn; distrib[15] = pinp
			return distrib
		if state == 8:
			distrib[9] = pinp*poutn; distrib[10] = pinp*poutp; distrib[13] = pinn*poutn
			distrib[14] = pinn*poutp
			return distrib
		if state == 9:
			distrib[9] = pinp*poutn; distrib[11] = pinp*poutp; distrib[13] = pinn*poutn
			distrib[15] = pinn*poutp
			return distrib
		if state == 10:
			distrib[10] = pinp*poutp; distrib[11] = pinp*poutn; distrib[14] = pinn*poutp
			distrib[15] = pinn*poutn
			return distrib
		if state == 11:
			distrib[11] = pinp; distrib[15] = pinn
			return distrib
		if state == 12:
			distrib[13] = poutn; distrib[14] = poutp
			return distrib
		if state == 13:
			distrib[13] = poutn; distrib[15] = poutp
			return distrib
		if state == 14:
			distrib[14] = poutp; distrib[15] = poutn
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (1, 2):
		if state == 0:
			distrib[5] = pinn*poutn**nv; distrib[6] = pinn*poutp**nv; distrib[7] = pinn*(-poutn**nv - poutp**nv + 1)
			distrib[9] = pinp*poutn**nv; distrib[10] = pinp*poutp**nv; distrib[11] = pinp*(-poutn**nv - poutp**nv + 1)
			return distrib
		if state == 1:
			distrib[5] = pinn*poutn**nv; distrib[7] = pinn*(-poutn**nv + 1); distrib[9] = pinp*poutn**nv
			distrib[11] = pinp*(-poutn**nv + 1)
			return distrib
		if state == 2:
			distrib[6] = pinn*poutp**nv; distrib[7] = pinn*(-poutp**nv + 1); distrib[10] = pinp*poutp**nv
			distrib[11] = pinp*(-poutp**nv + 1)
			return distrib
		if state == 3:
			distrib[7] = pinn; distrib[11] = pinp
			return distrib
		if state == 4:
			distrib[5] = pinn*poutn**nv; distrib[6] = pinn*poutp**nv; distrib[7] = pinn*(-poutn**nv - poutp**nv + 1)
			distrib[13] = pinp*poutn**nv; distrib[14] = pinp*poutp**nv; distrib[15] = pinp*(-poutn**nv - poutp**nv + 1)
			return distrib
		if state == 5:
			distrib[5] = pinn*poutn**nv; distrib[7] = pinn*(-poutn**nv + 1); distrib[13] = pinp*poutn**nv
			distrib[15] = pinp*(-poutn**nv + 1)
			return distrib
		if state == 6:
			distrib[6] = pinn*poutp**nv; distrib[7] = pinn*(-poutp**nv + 1); distrib[14] = pinp*poutp**nv
			distrib[15] = pinp*(-poutp**nv + 1)
			return distrib
		if state == 7:
			distrib[7] = pinn; distrib[15] = pinp
			return distrib
		if state == 8:
			distrib[9] = pinp*poutn**nv; distrib[10] = pinp*poutp**nv; distrib[11] = pinp*(-poutn**nv - poutp**nv + 1)
			distrib[13] = pinn*poutn**nv; distrib[14] = pinn*poutp**nv; distrib[15] = pinn*(-poutn**nv - poutp**nv + 1)
			return distrib
		if state == 9:
			distrib[9] = pinp*poutn**nv; distrib[11] = pinp*(-poutn**nv + 1); distrib[13] = pinn*poutn**nv
			distrib[15] = pinn*(-poutn**nv + 1)
			return distrib
		if state == 10:
			distrib[10] = pinp*poutp**nv; distrib[11] = pinp*(-poutp**nv + 1); distrib[14] = pinn*poutp**nv
			distrib[15] = pinn*(-poutp**nv + 1)
			return distrib
		if state == 11:
			distrib[11] = pinp; distrib[15] = pinn
			return distrib
		if state == 12:
			distrib[13] = poutn**nv; distrib[14] = poutp**nv; distrib[15] = -poutn**nv - poutp**nv + 1
			return distrib
		if state == 13:
			distrib[13] = poutn**nv; distrib[15] = -poutn**nv + 1
			return distrib
		if state == 14:
			distrib[14] = poutp**nv; distrib[15] = -poutp**nv + 1
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (2, 0):
		if state == 0:
			distrib[4] = pinn**nu; distrib[8] = pinp**nu; distrib[12] = -pinn**nu - pinp**nu + 1
			return distrib
		if state == 1:
			distrib[5] = pinn**nu; distrib[9] = pinp**nu; distrib[13] = -pinn**nu - pinp**nu + 1
			return distrib
		if state == 2:
			distrib[6] = pinn**nu; distrib[10] = pinp**nu; distrib[14] = -pinn**nu - pinp**nu + 1
			return distrib
		if state == 3:
			distrib[7] = pinn**nu; distrib[11] = pinp**nu; distrib[15] = -pinn**nu - pinp**nu + 1
			return distrib
		if state == 4:
			distrib[4] = pinn**nu; distrib[12] = -pinn**nu + 1
			return distrib
		if state == 5:
			distrib[5] = pinn**nu; distrib[13] = -pinn**nu + 1
			return distrib
		if state == 6:
			distrib[6] = pinn**nu; distrib[14] = -pinn**nu + 1
			return distrib
		if state == 7:
			distrib[7] = pinn**nu; distrib[15] = -pinn**nu + 1
			return distrib
		if state == 8:
			distrib[8] = pinp**nu; distrib[12] = -pinp**nu + 1
			return distrib
		if state == 9:
			distrib[9] = pinp**nu; distrib[13] = -pinp**nu + 1
			return distrib
		if state == 10:
			distrib[10] = pinp**nu; distrib[14] = -pinp**nu + 1
			return distrib
		if state == 11:
			distrib[11] = pinp**nu; distrib[15] = -pinp**nu + 1
			return distrib
		if state == 12:
			distrib[12] = 1
			return distrib
		if state == 13:
			distrib[13] = 1
			return distrib
		if state == 14:
			distrib[14] = 1
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (2, 1):
		if state == 0:
			distrib[5] = pinn**nu*poutn; distrib[6] = pinn**nu*poutp; distrib[9] = pinp**nu*poutn
			distrib[10] = pinp**nu*poutp; distrib[13] = poutn*(-pinn**nu - pinp**nu + 1); distrib[14] = poutp*(-pinn**nu - pinp**nu + 1)
			return distrib
		if state == 1:
			distrib[5] = pinn**nu*poutn; distrib[7] = pinn**nu*poutp; distrib[9] = pinp**nu*poutn
			distrib[11] = pinp**nu*poutp; distrib[13] = poutn*(-pinn**nu - pinp**nu + 1); distrib[15] = poutp*(-pinn**nu - pinp**nu + 1)
			return distrib
		if state == 2:
			distrib[6] = pinn**nu*poutp; distrib[7] = pinn**nu*poutn; distrib[10] = pinp**nu*poutp
			distrib[11] = pinp**nu*poutn; distrib[14] = poutp*(-pinn**nu - pinp**nu + 1); distrib[15] = poutn*(-pinn**nu - pinp**nu + 1)
			return distrib
		if state == 3:
			distrib[7] = pinn**nu; distrib[11] = pinp**nu; distrib[15] = -pinn**nu - pinp**nu + 1
			return distrib
		if state == 4:
			distrib[5] = pinn**nu*poutn; distrib[6] = pinn**nu*poutp; distrib[13] = poutn*(-pinn**nu + 1)
			distrib[14] = poutp*(-pinn**nu + 1)
			return distrib
		if state == 5:
			distrib[5] = pinn**nu*poutn; distrib[7] = pinn**nu*poutp; distrib[13] = poutn*(-pinn**nu + 1)
			distrib[15] = poutp*(-pinn**nu + 1)
			return distrib
		if state == 6:
			distrib[6] = pinn**nu*poutp; distrib[7] = pinn**nu*poutn; distrib[14] = poutp*(-pinn**nu + 1)
			distrib[15] = poutn*(-pinn**nu + 1)
			return distrib
		if state == 7:
			distrib[7] = pinn**nu; distrib[15] = -pinn**nu + 1
			return distrib
		if state == 8:
			distrib[9] = pinp**nu*poutn; distrib[10] = pinp**nu*poutp; distrib[13] = poutn*(-pinp**nu + 1)
			distrib[14] = poutp*(-pinp**nu + 1)
			return distrib
		if state == 9:
			distrib[9] = pinp**nu*poutn; distrib[11] = pinp**nu*poutp; distrib[13] = poutn*(-pinp**nu + 1)
			distrib[15] = poutp*(-pinp**nu + 1)
			return distrib
		if state == 10:
			distrib[10] = pinp**nu*poutp; distrib[11] = pinp**nu*poutn; distrib[14] = poutp*(-pinp**nu + 1)
			distrib[15] = poutn*(-pinp**nu + 1)
			return distrib
		if state == 11:
			distrib[11] = pinp**nu; distrib[15] = -pinp**nu + 1
			return distrib
		if state == 12:
			distrib[13] = poutn; distrib[14] = poutp
			return distrib
		if state == 13:
			distrib[13] = poutn; distrib[15] = poutp
			return distrib
		if state == 14:
			distrib[14] = poutp; distrib[15] = poutn
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
	if kind == (2, 2):
		if state == 0:
			distrib[5] = pinn**nu*poutn**nv; distrib[6] = pinn**nu*poutp**nv; distrib[7] = pinn**nu*(-poutn**nv - poutp**nv + 1)
			distrib[9] = pinp**nu*poutn**nv; distrib[10] = pinp**nu*poutp**nv; distrib[11] = pinp**nu*(-poutn**nv - poutp**nv + 1)
			distrib[13] = poutn**nv*(-pinn**nu - pinp**nu + 1); distrib[14] = poutp**nv*(-pinn**nu - pinp**nu + 1); distrib[15] = (pinn**nu + pinp**nu - 1)*(poutn**nv + poutp**nv - 1)
			return distrib
		if state == 1:
			distrib[5] = pinn**nu*poutn**nv; distrib[7] = pinn**nu*(-poutn**nv + 1); distrib[9] = pinp**nu*poutn**nv
			distrib[11] = pinp**nu*(-poutn**nv + 1); distrib[13] = poutn**nv*(-pinn**nu - pinp**nu + 1); distrib[15] = (poutn**nv - 1)*(pinn**nu + pinp**nu - 1)
			return distrib
		if state == 2:
			distrib[6] = pinn**nu*poutp**nv; distrib[7] = pinn**nu*(-poutp**nv + 1); distrib[10] = pinp**nu*poutp**nv
			distrib[11] = pinp**nu*(-poutp**nv + 1); distrib[14] = poutp**nv*(-pinn**nu - pinp**nu + 1); distrib[15] = (poutp**nv - 1)*(pinn**nu + pinp**nu - 1)
			return distrib
		if state == 3:
			distrib[7] = pinn**nu; distrib[11] = pinp**nu; distrib[15] = -pinn**nu - pinp**nu + 1
			return distrib
		if state == 4:
			distrib[5] = pinn**nu*poutn**nv; distrib[6] = pinn**nu*poutp**nv; distrib[7] = pinn**nu*(-poutn**nv - poutp**nv + 1)
			distrib[13] = poutn**nv*(-pinn**nu + 1); distrib[14] = poutp**nv*(-pinn**nu + 1); distrib[15] = (pinn**nu - 1)*(poutn**nv + poutp**nv - 1)
			return distrib
		if state == 5:
			distrib[5] = pinn**nu*poutn**nv; distrib[7] = pinn**nu*(-poutn**nv + 1); distrib[13] = poutn**nv*(-pinn**nu + 1)
			distrib[15] = (pinn**nu - 1)*(poutn**nv - 1)
			return distrib
		if state == 6:
			distrib[6] = pinn**nu*poutp**nv; distrib[7] = pinn**nu*(-poutp**nv + 1); distrib[14] = poutp**nv*(-pinn**nu + 1)
			distrib[15] = (pinn**nu - 1)*(poutp**nv - 1)
			return distrib
		if state == 7:
			distrib[7] = pinn**nu; distrib[15] = -pinn**nu + 1
			return distrib
		if state == 8:
			distrib[9] = pinp**nu*poutn**nv; distrib[10] = pinp**nu*poutp**nv; distrib[11] = pinp**nu*(-poutn**nv - poutp**nv + 1)
			distrib[13] = poutn**nv*(-pinp**nu + 1); distrib[14] = poutp**nv*(-pinp**nu + 1); distrib[15] = (pinp**nu - 1)*(poutn**nv + poutp**nv - 1)
			return distrib
		if state == 9:
			distrib[9] = pinp**nu*poutn**nv; distrib[11] = pinp**nu*(-poutn**nv + 1); distrib[13] = poutn**nv*(-pinp**nu + 1)
			distrib[15] = (pinp**nu - 1)*(poutn**nv - 1)
			return distrib
		if state == 10:
			distrib[10] = pinp**nu*poutp**nv; distrib[11] = pinp**nu*(-poutp**nv + 1); distrib[14] = poutp**nv*(-pinp**nu + 1)
			distrib[15] = (pinp**nu - 1)*(poutp**nv - 1)
			return distrib
		if state == 11:
			distrib[11] = pinp**nu; distrib[15] = -pinp**nu + 1
			return distrib
		if state == 12:
			distrib[13] = poutn**nv; distrib[14] = poutp**nv; distrib[15] = -poutn**nv - poutp**nv + 1
			return distrib
		if state == 13:
			distrib[13] = poutn**nv; distrib[15] = -poutn**nv + 1
			return distrib
		if state == 14:
			distrib[14] = poutp**nv; distrib[15] = -poutp**nv + 1
			return distrib
		if state == 15:
			distrib[15] = 1
			return distrib
